import torch 
import torch.nn as nn 
import torch.nn.functional as F
from transformers import Trainer 

class TinyBertForDistillation(nn.Module):
    def __init__(self, student_model, teacher_hidden_size=1024):
        super().__init__()
        self.bert = student_model
        self.config = self.bert.config 

        self.fit_dense_emb = nn.Linear(self.config.hidden_size, teacher_hidden_size)
        self.fit_dense_hidden = nn.Linear(self.config.hidden_size, teacher_hidden_size)

        nn.init.xavier_uniform_(self.fit_dense_emb.weight)
        nn.init.xavier_uniform_(self.fit_dense_hidden.weight)
    
    @property
    def device(self):
        """
        Restituisce il device su cui si trova il modello.
        Necessario perché il Trainer accede a model.device.
        """
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask, token_type_ids =None, labels=None, **kwargs):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )
    
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, alpha=0.5, temperature=5.0, weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.alpha = alpha
        self.temperature = temperature

        self.weights = weights if weights else {"pred": 1.0, "emb": 1.0, "hidden": 1.0, "attn": 1.0}

        self.teacher.eval() 
        for param in self.teacher.parameters():
            param.requires_grad = False 
        
        self.layer_mapping = {i: (i+1)*6 -1 for i in range(4)}
    
    def compute_loss(self, model, inputs, return_outputs = False, **kwagrs):
        if "labels" in inputs:
            labels = input.pop("labels")
        else:
            labels = None 

        inputs = {k: v.to(model.device) for k,v in inputs.items()}

        student_outputs = model(**inputs)

        student_logits      = student_outputs.logits.view(-1)
        student_hidden      = student_outputs.hidden_states
        student_attentions  = student_outputs.attentions

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs, output_hidden_states=True, output_attentions=True)
            teacher_logits      = teacher_outputs.logits.view(-1)
            teacher_hidden      = teacher_outputs.hidden_states 
            teacher_attentions  = teacher_outputs.attentions
        
        soft_target = torch.sigmoid(teacher_logits/self.temperature)
        soft_loss = F.binary_cross_entropy_with_logits(student_logits/self.temperature, soft_target) *(self.temperature **2)

        if labels is not None:
            hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels.float())
        else:
            hard_loss = 0.0

        loss_pred = (self.alpha * soft_loss) + ((1-self.alpha) * hard_loss)

        student_emb = model.fit_dense_emb(student_hidden[0])
        teacher_emb = teacher_hidden[0]
        loss_emb = F.mse_loss(student_emb, teacher_emb)

        loss_hidden = 0 
        for student_idx, teacher_idx in self.layer_mapping.items():
            student_feature = model.fit_dense_hidden(student_hidden[student_idx+1])
            teacher_feature = teacher_hidden[teacher_idx+1]
            loss_hidden += F.mse_loss(student_feature, teacher_feature) 
        
        loss_attn = 0 
        for student_idx, teacher_idx in self.layer_mapping.items():
            student_map = torch.mean(student_attentions[student_idx], dim=1)
            teacher_map = torch.mean(student_attentions[student_idx], dim=1)
            loss_attn += F.mse_loss(student_map, teacher_map)

        total_loss = (self.weights["pred"]*loss_pred) + (self.weights["emb"]*loss_emb) + (self.weights["hidden"] * loss_hidden) + (self.weights["attn"] * loss_attn)

        return (total_loss, student_outputs) if return_outputs else total_loss