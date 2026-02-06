import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import random
import os
import re
from tqdm import tqdm
import logging

# Configurazione Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prompt System per DeepSeek-R1
SYSTEM_PROMPT = """You are an expert solver of the NYT Connections game.
Your task is to receive 16 words and group them into 4 precise semantic groups of 4 words each.
BE CAREFUL: some words have double senses!

STRICT RULES:
1. First, analyze the relationships between the words step by step in your thinking process.
2. The final output must be a valid JSON object ONLY, containing the 4 groups.
3. The format must be exactly:
{
  "groups": [
    {"words": ["W1", "W2", "W3", "W4"], "description": "CATEGORY NAME"},
    ...
  ]
}
"""

def setup_teacher_model(model_id):
    """
    Carica il Teacher (DeepSeek) in 4-bit per risparmiare VRAM.
    """
    logger.info(f"Loading Teacher Model: {model_id}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        attn_implementation="eager" # Eager per stabilità in generazione
    )
    
    return model, tokenizer

def calculate_w_x(teacher_json, ground_truth):
    """
    Calcola il peso w(x) basato sulla correttezza (0.0, 0.25, 0.5, 0.75, 1.0).
    """
    try:
        if not teacher_json or 'groups' not in teacher_json:
            return 0.0
        
        pred_groups = [set(g['words']) for g in teacher_json['groups']]
        true_groups = [set(a['words']) for a in ground_truth['answers']]
        
        correct_count = 0
        for pg in pred_groups:
            if pg in true_groups:
                correct_count += 1
        
        return correct_count / 4.0
    except Exception as e:
        return 0.0

def extract_json_from_text(text):
    """
    Tenta di estrarre il blocco JSON dall'output del modello usando regex.
    """
    try:
        # Cerca il primo blocco tra parentesi graffe
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except:
        pass
    return {}

def generate_augmented_dataset(
    input_file, 
    output_file, 
    teacher_model_id, 
    augmentations_per_puzzle=3, 
    batch_save_interval=10
):
    """
    Funzione principale che orchestra la generazione dei dati.
    """
    
    if os.path.exists(output_file):
        logger.warning(f"File {output_file} already exists. Appending to it (or delete manually to restart).")

    # 1. Carica Modello
    model, tokenizer = setup_teacher_model(teacher_model_id)
    
    # 2. Carica Dataset Input
    with open(input_file, 'r', encoding='utf-8') as f:
        puzzles = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(puzzles)} puzzles from {input_file}")
    
    buffer = []
    total_generated = 0
    
    # Apriamo in append mode per non perdere progressi se crasha
    with open(output_file, 'a', encoding='utf-8') as f_out:
        
        for p_idx, puzzle in enumerate(tqdm(puzzles, desc="Generating Data")):
            original_words = puzzle['words']
            
            # Augmentation Loop
            for i in range(augmentations_per_puzzle):
                # Shuffle
                current_words = original_words.copy()
                random.shuffle(current_words)
                
                # Prompt
                words_str = ", ".join(current_words)
                user_content = f"Classify these 16 words into 4 groups: {words_str}"
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ]
                
                input_ids = tokenizer.apply_chat_template(
                    messages, 
                    return_tensors="pt", 
                    add_generation_prompt=True
                ).to(model.device)
                
                # Inference
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids, 
                        max_new_tokens=1500,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decoding
                # skip_special_tokens=False per vedere eventuali token di controllo se servono,
                # ma rimuoviamo il prompt dall'output
                generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
                clean_text = generated_text.replace("<|eot_id|>", "").strip()
                
                # Parsing & Scoring
                teacher_json = extract_json_from_text(clean_text)
                w_x = calculate_w_x(teacher_json, puzzle)
                
                # Create Entry
                entry = {
                    "puzzle_id": puzzle['puzzle_id'],
                    "aug_id": i,
                    "shuffled_input": current_words,
                    # Salviamo il prompt formattato per debug/uso futuro
                    "prompt_text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                    "teacher_output_text": clean_text, # Include <think> e JSON
                    "teacher_json": teacher_json,
                    "ground_truth": puzzle['answers'],
                    "w_x": w_x
                }
                
                buffer.append(entry)
                total_generated += 1
            
            # Batch Saving
            if (p_idx + 1) % batch_save_interval == 0:
                for item in buffer:
                    f_out.write(json.dumps(item) + "\n")
                buffer = []
                f_out.flush()
        
        # Save remaining
        if buffer:
            for item in buffer:
                f_out.write(json.dumps(item) + "\n")
                
    logger.info(f"Generation Complete. Total examples: {total_generated}. Saved to {output_file}")