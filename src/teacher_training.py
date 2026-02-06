import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeneralDistillation:
    def __init__(self, student_model_id, teacher_model_id, output_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir

        #Loading Teacher (4-bit, frozen weights)
        logger.info("Loading Teacher (DeepSeek)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_id, quantization_config = bnb_config, 
            device_map={"": 0},
            attn_implementation= "eager",
            torch_dtype=torch.bfloat16,
            output_hidden_states = True,
            output_attentions=True
        )
        self.teacher.eval()

        #Loading Student 
        logger.info("Loading Student (LLama 3.2 1B)...")
        self.student = AutoModelForCausalLM.from_pretrained(
            student_model_id,
            torch_dtype=torch.bfloat16, 
            output_hidden_states=True,
            output_attentions=True
        ).to(self.device)
        self.teacher.config.use_cache = False
        self.student.config.use_cache = False


        #Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token 

        #Layer mapping (Uniform: g(m) = 2m)
        #Because student has 16 layers while teacher has 32 layers
        self.layer_mapping = [2*layer for layer in range(self.student.config.num_hidden_layers)]
        self.layer_mapping.append(self.teacher.config.num_hidden_layers -1)

        #Projection Matrix
        #Student size = 2048  Teacher size = 4096
        self.student_dim = self.student.config.hidden_size 
        self.teacher_dim = self.teacher.config.hidden_size 

        self.projector = nn.Linear(self.student_dim, self.teacher_dim, bias=False).to(self.device, dtype=torch.bfloat16)

        self.optimizer = torch.optim.AdamW(
            list(self.student.parameters()) + list(self.projector.parameters()), lr = 5e-5
        )
        self.mse_loss = nn.MSELoss()
    
    def calculate_loss(self, student_outputs, teacher_outputs):
        total_loss = 0.0

        #Embedding layer Distillation 
        student_emb = student_outputs.hidden_states[0]
        teacher_emb = teacher_outputs.hidden_states[0].to(dtype=torch.bfloat16)
        proj_student_emb = self.projector(student_emb)
        loss_emb = self.mse_loss(proj_student_emb, teacher_emb)
        total_loss += loss_emb

        #Hidden States Distillation
        student_hidden = student_outputs.hidden_states[1:]
        teacher_hidden = teacher_outputs.hidden_states[1:]

        loss_hidden = 0.0
        for i, student_idx in enumerate(range(len(student_hidden))):
            if i < len(self.layer_mapping):
                teacher_idx = self.layer_mapping[i]
                proj_student_state = self.projector(student_hidden[student_idx])
                target_state = teacher_hidden[teacher_idx].to(dtype=torch.bfloat16)
                loss_hidden += self.mse_loss(proj_student_state, target_state)
        
        total_loss += loss_hidden 

        #Attention Matrices Distillation 
        student_attention = student_outputs.attentions
        teacher_attention = teacher_outputs.attentions 

        loss_attn = 0.0
        for i, student_idx in enumerate(range(len(student_attention))):
            teacher_idx = self.layer_mapping[i]
            target_attn = teacher_attention[teacher_idx].to(dtype=torch.bfloat16)
            loss_attn += self.mse_loss(student_attention[student_idx], target_attn)
        total_loss += loss_attn

        return total_loss
    
    def train(self, train_file, epochs=1, batch_size=2,accumulation_steps=8):
        dataset = load_dataset("json", data_files={"train": train_file})["train"]
        def collate_fn(batch):
            texts = [item['full_text'] for item in batch]
            return self.tokenizer(
                texts,
                return_tensors="pt",
                padding = True,
                truncation=True,
                max_length=128
            )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        logger.info("Starting General Distillation...")

        for epoch in range(epochs):
            self.student.train()
            epoch_loss = 0.0 
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

            for step, batch in enumerate(progress_bar):
                inputs = {k: v.to(self.device) for k,v in batch.items()}

                #1. Forward Teacher (No Grad)
                with torch.no_grad():
                    teacher_outputs = self.teacher(**inputs)
                
                #2. Forward Student (Grad)
                student_outputs = self.student(**inputs)

                #3. Loss calculation
                loss = self.calculate_loss(student_outputs, teacher_outputs)
                loss = loss / accumulation_steps

                '''
                if torch.isnan(loss):
                    print(f"Skipping Step {step}: Loss is NaN!")
                    self.optimizer.zero_grad()
                    continue
                '''
                loss.backward()

                if (step+1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.projector.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                epoch_loss += loss.item() * accumulation_steps
                progress_bar.set_postfix({"loss": loss.item() * accumulation_steps})
            
            logger.info(f"Epoch {epoch+1} completed. Avg loss; {epoch_loss/len(dataloader)}")

            save_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}")
            self.student.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            torch.save(self.projector.state_dict(), os.path.join(save_path, "projector.pt"))
        
        logger.info("General Distillation DONE")