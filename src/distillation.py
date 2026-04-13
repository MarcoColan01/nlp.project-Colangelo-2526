import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
from types import MethodType
from transformers import Trainer 

def unnormalized_attention_forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, **kwargs):
    
    def transpose_for_scores(x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    mixed_query_layer = self.query(hidden_states)

    key_layer = transpose_for_scores(self.key(hidden_states))
    value_layer = transpose_for_scores(self.value(hidden_states))
    query_layer = transpose_for_scores(mixed_query_layer)

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores /= math.sqrt(self.attention_head_size) 

    if attention_mask is not None:
        attention_scores += attention_mask

    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    attention_probs_dropped = self.dropout(attention_probs)
    
    context_layer = torch.matmul(attention_probs_dropped, value_layer)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous() 
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) 
    context_layer = context_layer.view(*new_context_layer_shape) 

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

    s = torch.nan_to_num(s, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
    t = torch.nan_to_num(t, nan=0.0, posinf=clamp_val, neginf=-clamp_val)

    if clamp_val is not None:
        s = s.clamp(-clamp_val, clamp_val)
        t = t.clamp(-clamp_val, clamp_val)

    q_mask = attention_mask[:, None, :, None].bool()   
    k_mask = attention_mask[:, None, None, :].bool()   
    pair_mask = q_mask & k_mask                        

    s = s.masked_fill(~pair_mask, 0.0)
    t = t.masked_fill(~pair_mask, 0.0)

    diff2 = (s - t).pow(2)
    diff2 = diff2 * pair_mask.to(diff2.dtype)  

    denom = (pair_mask.sum().to(diff2.dtype) * s.size(1)).clamp_min(1.0)
    return diff2.sum() / denom

class Phase2DistillationTrainer(Trainer):
    def __init__(
            self, 
            teacher_model,
            *args,
            temperature=1.0,
            rho_ok=0.9,
            rho_bad=0.2,
            class_weights=None,
            debug_finite_check=True,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval() 
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        self.T = temperature
        self.rho_ok = rho_ok
        self.rho_bad = rho_bad
        self.class_weights = class_weights  
        self.debug_finite_check = debug_finite_check
        self._last_td2_logs = {} 
    
    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        labels = inputs["labels"].long() 
        student_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        with torch.no_grad():
            teacher_outputs = self.teacher(**student_inputs, return_dict=True)
            z_T = teacher_outputs.logits  
        
        student_outputs = model(**student_inputs, return_dict=True)
        z_S = student_outputs.logits  

        teacher_preds = z_T.argmax(dim=-1)  
        correct_mask = (teacher_preds == labels)  
        rho = torch.where(correct_mask,
                          torch.full_like(labels, self.rho_ok, dtype=torch.float), 
                          torch.full_like(labels, self.rho_bad, dtype=torch.float))  
        
        soft_student = F.log_softmax(z_S / self.T, dim=-1)  
        soft_teacher = F.softmax(z_T / self.T, dim=-1)  
        kl_per_sample = F.kl_div(soft_student, soft_teacher, reduction='none').sum(dim=-1)  
        soft_loss_per_sample = kl_per_sample * (self.T **2) 
        soft_loss = (rho * soft_loss_per_sample).mean()

        if self.class_weights is not None:
            w = self.class_weights.to(z_S.device)
            hard_loss_fct = nn.CrossEntropyLoss(weight=w, reduction='none')
        else:
            hard_loss_fct = nn.CrossEntropyLoss(reduction='none')
        hard_loss_per_sample = hard_loss_fct(z_S, labels)  
        hard_loss = ((1-rho) * hard_loss_per_sample).mean()

        total_loss = soft_loss + hard_loss

        self._last_td2_logs = {
            "td2_loss_soft": float(soft_loss.detach().cpu()),
            "td2_loss_hard": float(hard_loss.detach().cpu()),
            "tds_rho_mean": float(rho.mean().detach().cpu()),
            "td2_loss_total": float(total_loss.detach().cpu()),
        }

        if self.debug_finite_check and (not torch.isfinite(total_loss)):
            raise FloatingPointError(f"Non-finite TD2 loss. "
                                     f"soft={soft_loss.item():.6f}, hard={hard_loss.item():.6f}")

        return (total_loss, student_outputs) if return_outputs else total_loss
    
    def log(self, logs, *args, **kwargs):
        if isinstance(logs, dict) and self._last_td2_logs:
            logs = {**logs, **self._last_td2_logs}
        return super().log(logs, *args, **kwargs)

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
        self._last_td_logs = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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

        mapping = [0, 3, 6, 9, 12]

        loss_embd = torch.zeros((), device=attention_mask.device, dtype=torch.float32)
        loss_hidn = torch.zeros((), device=attention_mask.device, dtype=torch.float32)
        loss_attn = torch.zeros((), device=attention_mask.device, dtype=torch.float32)

        hidn_count = 0
        attn_count = 0

        for student_idx, teacher_idx in enumerate(mapping):
            proj_s = model.fit_dense[student_idx](student_hidden[student_idx])

            if student_idx == 0:
                loss_embd = _masked_hidden_mse(proj_s, teacher_hidden[teacher_idx], attention_mask)
            else:
                loss_hidn = loss_hidn + _masked_hidden_mse(proj_s, teacher_hidden[teacher_idx], attention_mask)
                hidn_count += 1

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

        self._last_td_logs = {
            "td_loss_embd": float(loss_embd.detach().cpu()),
            "td_loss_hidn": float(loss_hidn.detach().cpu()) if torch.is_tensor(loss_hidn) else float(loss_hidn),
            "td_loss_attn": float(loss_attn.detach().cpu()) if torch.is_tensor(loss_attn) else float(loss_attn),
            "td_loss_total_raw": float(loss.detach().cpu()),
        }

        if self.debug_finite_check and (not torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite TD1 loss. "
                f"embd={loss_embd.item():.6f}, hidn={loss_hidn.item():.6f}, attn={loss_attn.item():.6f}"
            )

        return (loss, student_outputs) if return_outputs else loss
    
    def log(self, logs, *args, **kwargs):
        if isinstance(logs, dict) and self._last_td_logs:
            logs = {**logs, **self._last_td_logs}
        return super().log(logs, *args, **kwargs)