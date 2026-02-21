import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
from types import MethodType
from transformers import Trainer 

def unnormalized_attention_forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, **kwargs):
    
    # 1. Definiamo noi la funzione di trasposizione che HF ha rimosso
    def transpose_for_scores(x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 2. Calcoliamo query, key e value
    mixed_query_layer = self.query(hidden_states)

    key_layer = transpose_for_scores(self.key(hidden_states))
    value_layer = transpose_for_scores(self.value(hidden_states))
    query_layer = transpose_for_scores(mixed_query_layer)

    # 3. Calcolo dell'Attention Score crudo (PRE-SOFTMAX)
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores /= math.sqrt(self.attention_head_size) 

    if attention_mask is not None:
        attention_scores += attention_mask

    # 4. Softmax standard per continuare il forward pass
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_dropped = self.dropout(attention_probs)
    
    context_layer = torch.matmul(attention_probs_dropped, value_layer)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous() 
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) 
    context_layer = context_layer.view(*new_context_layer_shape) 

    # 5. IL TRUCCO: Se ci chiedono output_attentions, restituiamo gli SCORES CRUDI, non le probabilità
    outputs = (context_layer, attention_scores) if output_attentions else (context_layer,)
    return outputs

def patch_model_for_unnormalized_attention(model):
    for layer in model.bert.encoder.layer:
        layer.attention.self.forward = MethodType(unnormalized_attention_forward, layer.attention.self)
    print(f"Model {model.config.model_type} patched: now output_attentions returns pre-softmax logits.")



class TinyBERTPhase1(nn.Module):
    def __init__(self, student_model, teacher_hidden_size=768, student_hidden_size=312):
        super().__init__()
        self.student = student_model

        if hasattr(student_model, 'classifier'):
            for param in self.student.classifier.parameters():
                param.requires_grad = False 
        
        self.fit_dense = nn.ModuleList([
            nn.Linear(student_hidden_size, teacher_hidden_size) for _ in range(5)
        ])
        

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        outputs=self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
            output_attentions=True
        ) 
        return outputs

def _masked_hidden_mse(student_h, teacher_h, attention_mask):
    """
    student_h, teacher_h: [B, L, H]
    attention_mask: [B, L]
    """
    s = student_h.float()
    t = teacher_h.float()

    token_mask = attention_mask.unsqueeze(-1).to(s.dtype)  # [B, L, 1]
    diff2 = (s - t).pow(2) * token_mask

    denom = (token_mask.sum() * s.size(-1)).clamp_min(1.0)
    return diff2.sum() / denom


def _masked_attn_mse(student_a, teacher_a, attention_mask, clamp_val=30.0):
    """
    student_a, teacher_a: [B, heads, L, L]  (pre-softmax logits)
    attention_mask: [B, L]
    """
    s = student_a.float()
    t = teacher_a.float()

    # Sanitizzazione preventiva (utile se compare nan/inf)
    s = torch.nan_to_num(s, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
    t = torch.nan_to_num(t, nan=0.0, posinf=clamp_val, neginf=-clamp_val)

    # Clamp dei logits validi per stabilità numerica in MSE
    if clamp_val is not None:
        s = s.clamp(-clamp_val, clamp_val)
        t = t.clamp(-clamp_val, clamp_val)

    # Mask bidirezionale: consideriamo solo query/key reali (no padding)
    q_mask = attention_mask[:, None, :, None].bool()   # [B,1,L,1]
    k_mask = attention_mask[:, None, None, :].bool()   # [B,1,1,L]
    pair_mask = q_mask & k_mask                        # [B,1,L,L]

    s = s.masked_fill(~pair_mask, 0.0)
    t = t.masked_fill(~pair_mask, 0.0)

    diff2 = (s - t).pow(2)
    diff2 = diff2 * pair_mask.to(diff2.dtype)  # broadcast sulla head dim

    denom = (pair_mask.sum().to(diff2.dtype) * s.size(1)).clamp_min(1.0)
    return diff2.sum() / denom


class Phase1DistillationTrainer(Trainer):
    def __init__(
        self,
        teacher_model,
        *args,
        lambda_embd=1.0,
        lambda_hidn=1.0,
        lambda_attn=1.0,
        attn_clamp_val=30.0,
        debug_finite_check=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.lambda_embd = lambda_embd
        self.lambda_hidn = lambda_hidn
        self.lambda_attn = lambda_attn
        self.attn_clamp_val = attn_clamp_val
        self.debug_finite_check = debug_finite_check

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # labels non servono in TD-1
        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        attention_mask = teacher_inputs["attention_mask"]

        with torch.no_grad():
            teacher_outputs = self.teacher(
                **teacher_inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        student_outputs = model(**inputs)

        teacher_hidden = teacher_outputs.hidden_states
        teacher_attn = teacher_outputs.attentions
        student_hidden = student_outputs.hidden_states
        student_attn = student_outputs.attentions

        # TinyBERT 4L -> BERT-base 12L (emb + 4 hidden mapping)
        mapping = [0, 3, 6, 9, 12]

        loss_embd = torch.zeros((), device=attention_mask.device, dtype=torch.float32)
        loss_hidn = torch.zeros((), device=attention_mask.device, dtype=torch.float32)
        loss_attn = torch.zeros((), device=attention_mask.device, dtype=torch.float32)

        hidn_count = 0
        attn_count = 0

        for student_idx, teacher_idx in enumerate(mapping):
            proj_s = model.fit_dense[student_idx](student_hidden[student_idx])

            if student_idx == 0:
                # embedding distillation
                loss_embd = _masked_hidden_mse(proj_s, teacher_hidden[teacher_idx], attention_mask)
            else:
                loss_hidn = loss_hidn + _masked_hidden_mse(proj_s, teacher_hidden[teacher_idx], attention_mask)
                hidn_count += 1

                # attenzione: student layer (0..3) <-> teacher layer (2,5,8,11)
                s_a = student_attn[student_idx - 1]
                t_a = teacher_attn[teacher_idx - 1]
                loss_attn = loss_attn + _masked_attn_mse(s_a, t_a, attention_mask, clamp_val=self.attn_clamp_val)
                attn_count += 1

        if hidn_count > 0:
            loss_hidn = loss_hidn / hidn_count
        if attn_count > 0:
            loss_attn = loss_attn / attn_count

        loss = (
            self.lambda_embd * loss_embd +
            self.lambda_hidn * loss_hidn +
            self.lambda_attn * loss_attn
        )

        if self.debug_finite_check and (not torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite TD1 loss. "
                f"embd={loss_embd.item():.6f}, hidn={loss_hidn.item():.6f}, attn={loss_attn.item():.6f}"
            )

        return (loss, student_outputs) if return_outputs else loss