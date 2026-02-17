import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import Trainer 

def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff2 = (a-b)**2 
    diff2 = diff2*mask 
    return diff2.sum() / (mask.sum().clamp_min(eps))

def make_token_mask(attention_mask: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    m = attention_mask.float().unsqueeze(-1)
    return m 

def make_attn_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    m = attention_mask.float() 
    return (m[:, None, :, None]* m[:,None,None,:])

def pool_teacher_heads(attn_t: torch.Tensor) -> torch.Tensor:
    B,H,L,_ =  attn_t.shape 
    assert H==16
    return attn_t.view(B,8,2,L,L).mean(dim=2)

class StudentWithProjections(nn.Module):
    def __init__(self, student_module: nn.Module, student_hidden: int = 512, teacher_hidden: int = 1024):
        super().__init__()
        self.student = student_module
        self.proj_emb = nn.Linear(student_hidden, teacher_hidden, bias=False)
        self.proj_hid = nn.Linear(student_hidden, teacher_hidden, bias=False)
        nn.init.xavier_uniform_(self.proj_emb.weight)
        nn.init.xavier_uniform_(self.proj_hid.weight)

    @property 
    def device(self):
        return next(self.parameters()).device 

    def forward(self, **inputs):
        return self.student(
            **inputs,
            output_hidden_states=True,
            output_attentions=True, 
            return_dict=True
        ) 
    
class TD1Trainer(Trainer):
    def __init__(
            self, 
            teacher_model: nn.Module,
            layer_mapping: dict[int, int] | None = None,
            lambda_emb: float = 1.0,
            lambda_hid: float = 1.0,
            lambda_attn: float = 0.5,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.label_names = ["labels"]
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        self.layer_mapping = layer_mapping or {0: 5, 1: 11, 2: 17, 3: 23}
        self.lambda_emb = lambda_emb
        self.lambda_hid = lambda_hid
        self.lambda_attn = lambda_attn
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels", None)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        attention_mask = inputs.get("attention_mask", None)

        if next(self.teacher.parameters()).device != model.device:
            self.teacher.to(model.device)

        student_out = model(**inputs)

        with torch.no_grad():
            teacher_out = self.teacher(
                **inputs, 
                output_hidden_states=True,
                output_attentions=True,
                return_dict = True
            )
        
        student_emb = model.proj_emb(student_out.hidden_states[0])
        teacher_emb = teacher_out.hidden_states[0]
        if attention_mask is not None:
            tok_mask = make_token_mask(attention_mask, student_emb.size(-1))
            loss_emb = masked_mse(student_emb, teacher_emb, tok_mask)
        else:
            loss_emb = F.mse_loss(student_emb, teacher_emb)
        
        loss_hid = 0.0 
        for student_idx, teacher_idx in self.layer_mapping.items():
            student_hidden = model.proj_hid(student_out.hidden_states[student_idx+1])
            teacher_hidden = teacher_out.hidden_states[teacher_idx+1]

            if attention_mask is not None:
                tok_mask = make_token_mask(attention_mask, student_hidden.size(-1))
                loss_hid += masked_mse(student_hidden, teacher_hidden, tok_mask)
            else:
                loss_hid += F.mse_loss(student_hidden, teacher_hidden)
        
        loss_attn = 0.0 
        for student_idx, teacher_idx in self.layer_mapping.items():
            student_attn = student_out.attentions[student_idx]
            teacher_attn = teacher_out.attentions[teacher_idx]
            teacher_attn_pooled = pool_teacher_heads(teacher_attn)

            if attention_mask is not None:
                attn_mask = make_attn_mask(attention_mask)
                loss_attn += masked_mse(student_attn, teacher_attn_pooled, attn_mask)
            else:
                loss_attn += F.mse_loss(student_attn, teacher_attn_pooled)
        
        total = (
            self.lambda_emb*loss_emb+
            self.lambda_hid*loss_hid+
            self.lambda_attn*loss_attn
        )

        if return_outputs:
            return total, student_out.logits   # <-- SOLO logits
        return total