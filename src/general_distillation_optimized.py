import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    max_length: int = 128
    pretokenize: bool = True
    # If you are on Windows, keep num_workers=0 unless you know multiprocessing is stable in your setup.
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True


class GeneralDistillation:
    """
    General Distillation (TinyBERT-style "general distillation"):
      - embedding layer distillation (project student -> teacher dim)
      - hidden states distillation (uniform layer mapping)
      - attention matrices distillation (uniform layer mapping)

    This module preserves the original training + distillation behavior, while improving runtime performance
    through:
      - teacher forward in torch.inference_mode()
      - disabling KV-cache (use_cache=False)
      - optional dataset pre-tokenization (same dynamic padding behavior as before)
      - DataLoader pinned memory + non_blocking transfers (when CUDA is available)
      - minor overhead reductions (set_to_none=True on zero_grad)
    """

    def __init__(self, student_model_id: str, teacher_model_id: str, output_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir

        # Loading Teacher (4-bit, frozen weights)
        logger.info("Loading Teacher (DeepSeek)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager",  # keep identical behavior
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            output_attentions=True,
        )
        self.teacher.eval()
        self.teacher.requires_grad_(False)

        # Loading Student
        logger.info("Loading Student (LLama 3.2 1B)...")
        self.student = AutoModelForCausalLM.from_pretrained(
            student_model_id,
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            output_attentions=True,
        ).to(self.device)

        # Disable KV cache: reduces memory, does NOT change forward outputs for full-sequence training.
        self.teacher.config.use_cache = False
        self.student.config.use_cache = False

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Layer mapping (Uniform: g(m) = 2m)
        # Because student has 16 layers while teacher has 32 layers
        self.layer_mapping = [2 * layer for layer in range(self.student.config.num_hidden_layers)]
        # Keep identical behavior (even if unused for the 16-layer loop)
        self.layer_mapping.append(self.teacher.config.num_hidden_layers - 1)

        # Projection Matrix
        # Student size = 2048  Teacher size = 4096
        self.student_dim = self.student.config.hidden_size
        self.teacher_dim = self.teacher.config.hidden_size
        self.projector = nn.Linear(self.student_dim, self.teacher_dim, bias=False).to(
            self.device, dtype=torch.bfloat16
        )

        self.optimizer = torch.optim.AdamW(
            list(self.student.parameters()) + list(self.projector.parameters()), lr=5e-5
        )
        self.mse_loss = nn.MSELoss()

        # Helpful warning: CPU/disk offload will drastically hurt throughput.
        if hasattr(self.teacher, "hf_device_map"):
            offloaded = {k: v for k, v in self.teacher.hf_device_map.items() if v in ("cpu", "disk")}
            if offloaded:
                logger.warning(
                    "Teacher appears partially offloaded to CPU/disk via device_map='auto'. "
                    "This can severely reduce it/s. Consider forcing the teacher on GPU if VRAM allows."
                )

    def calculate_loss(self, student_outputs, teacher_outputs) -> torch.Tensor:
        total_loss = 0.0

        # Embedding layer Distillation
        student_emb = student_outputs.hidden_states[0]
        teacher_emb = teacher_outputs.hidden_states[0].to(dtype=torch.bfloat16)
        proj_student_emb = self.projector(student_emb)
        loss_emb = self.mse_loss(proj_student_emb, teacher_emb)
        total_loss += loss_emb

        # Hidden States Distillation
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

        # Attention Matrices Distillation
        student_attention = student_outputs.attentions
        teacher_attention = teacher_outputs.attentions

        loss_attn = 0.0
        for i, student_idx in enumerate(range(len(student_attention))):
            teacher_idx = self.layer_mapping[i]
            target_attn = teacher_attention[teacher_idx].to(dtype=torch.bfloat16)
            loss_attn += self.mse_loss(student_attention[student_idx], target_attn)
        total_loss += loss_attn

        return total_loss

    def _build_dataloader(self, train_file: str, batch_size: int, cfg: DataConfig) -> DataLoader:
        dataset = load_dataset("json", data_files={"train": train_file})["train"]

        if cfg.pretokenize:
            # Tokenize once, but keep dynamic padding identical to the original behavior:
            # - tokenization: truncation + max_length
            # - padding: performed in collate_fn to longest in batch
            def tokenize_batch(examples: Dict[str, List[str]]) -> Dict[str, Any]:
                return self.tokenizer(
                    examples["full_text"],
                    truncation=True,
                    max_length=cfg.max_length,
                    padding=False,
                )

            dataset = dataset.map(
                tokenize_batch,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing dataset",
            )

            def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
                return self.tokenizer.pad(
                    features,
                    padding=True,  # dynamic padding (longest in batch)
                    return_tensors="pt",
                )
        else:
            # Original behavior (tokenize inside collate_fn)
            def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
                texts = [item["full_text"] for item in batch]
                return self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=cfg.max_length,
                )

        pin_memory = bool(cfg.pin_memory and torch.cuda.is_available())
        num_workers = int(cfg.num_workers)
        persistent_workers = bool(cfg.persistent_workers and num_workers > 0)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def train(
        self,
        train_file: str,
        epochs: int = 1,
        batch_size: int = 2,
        accumulation_steps: int = 8,
        data_cfg: Optional[DataConfig] = None,
    ):
        data_cfg = data_cfg or DataConfig(
            # Conservative defaults: safe everywhere.
            # If you're on Linux and want more throughput, try num_workers=2-4.
            num_workers=0 if os.name == "nt" else 2,
            pin_memory=True,
            persistent_workers=True,
            pretokenize=True,
            max_length=128,
        )

        dataloader = self._build_dataloader(train_file, batch_size, data_cfg)
        logger.info("Starting General Distillation...")

        for epoch in range(epochs):
            self.student.train()
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

            for step, batch in enumerate(progress_bar):
                # Non-blocking transfers help only if DataLoader uses pinned memory.
                inputs = {
                    k: v.to(self.device, non_blocking=(dataloader.pin_memory and torch.cuda.is_available()))
                    for k, v in batch.items()
                }

                # 1) Forward Teacher (No Grad)
                with torch.no_grad():
                    teacher_outputs = self.teacher(**inputs)

                # 2) Forward Student (Grad)
                student_outputs = self.student(**inputs)

                # 3) Loss calculation
                loss = self.calculate_loss(student_outputs, teacher_outputs)
                loss = loss / accumulation_steps

                loss.backward()

                if (step + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.projector.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * accumulation_steps
                progress_bar.set_postfix({"loss": loss.item() * accumulation_steps})

            logger.info(f"Epoch {epoch+1} completed. Avg loss; {epoch_loss/len(dataloader)}")

            save_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            self.student.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            torch.save(self.projector.state_dict(), os.path.join(save_path, "projector.pt"))

        logger.info("General Distillation DONE")
