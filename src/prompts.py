'''
Obiettivi del modulo:
    - costruire in modo deterministico l’input a partire da words + answers[*].answerDescription (categorie),
    - imporre un formato di output rigidissimo (JSON) per ridurre al minimo problemi di parsing,
    - usare una sola tecnica di prompt engineering (single-call “method-actor light” in un’unica generazione, senza multi-call),
    - evitare special tokens: testo “plain”, compatibile con il tuo setup add_special_tokens=False.
'''

from __future__ import annotations
from dataclasses import dataclass 
from typing import List, Dict, Any, Sequence 
import json 
import random 

@dataclass(frozen=True)
class ConnectionsPromptConfig:
    max_reasoning_words: int = 120
    forbid_extra_words: bool = True
    output_json_only: bool = True 
    seed: int = 0
    shuffle_words: bool = False
    shuffle_categories: bool = False

def normalize_word(w: str) -> str: 
    return " ".join(w.strip().split())

def build_task_input(
        words: Sequence[str],
        categories: Sequence[str],
        config: ConnectionsPromptConfig,
) -> str:
    words_n = [normalize_word(w) for w in words]
    categories_n = [normalize_word(c) for c in categories]

    rnd = random.Random(config.seed)

    if config.shuffle_words:
        words_n = words_n[:]
        rnd.shuffle(words_n)

    if config.shuffle_categories:
        categories_n = categories_n[:]
        rnd.shuffle(categories_n)
    
    lines = []
    lines.append("NYT Connections (categories are provided).")
    lines.append("")
    lines.append("Words (16):")
    for i,w in enumerate(words_n, start=1):
        lines.append(f"{i:02d}. {w}")
    lines.append("")
    lines.append("Categories (4):") 
    for i,c in enumerate(categories_n, start=1):
        lines.append(f"{i}. {c}")
    return "\n".join(lines)

def _system_instruction(cfg: ConnectionsPromptConfig) -> str:

    constraints = []
    constraints.append("You must assign all 16 words to exactly one of the 4 given categories.")
    constraints.append("Each category must contain exactly 4 words.")
    constraints.append("Do not invent new words.")
    constraints.append("Do not repeat words across categories.")
    if cfg.forbid_extra_words:
        constraints.append("Use only the exact words provided (case-insensitive match is allowed).")

    # output rigido: solo JSON
    if cfg.output_json_only:
        constraints.append("Output must be valid JSON and nothing else.")

    # reasoning breve: ma *dentro* il JSON per evitare parsing di testo libero
    # (Così non devi parsare <THINK> ecc. e resti robusto)
    constraints.append(
        f'Include a brief explanation (<= {cfg.max_reasoning_words} words) '
        'as field "reasoning" inside the JSON.'
    )

    return " ".join(constraints)

def build_teacher_prompt(
        words: Sequence[str],
        categories: Sequence[str],
        cfg: ConnectionsPromptConfig,
) -> str:
    input_block = build_task_input(words, categories, cfg)
    instr = _system_instruction(cfg)
    schema = {
        "reasoning": "string (brief explanation)",
        "groups": [
            {"category": "string", "words": ["w1", "w2", "w3", "w4"]},
            {"category": "string", "words": ["w1", "w2", "w3", "w4"]},
            {"category": "string", "words": ["w1", "w2", "w3", "w4"]},
            {"category": "string", "words": ["w1", "w2", "w3", "w4"]},
        ],
    }

    prompt = (
        f"{instr}\n\n"
        f"{input_block}\n\n"
        "Think step-by-step internally, but DO NOT output your internal steps.\n"
        "Before finalizing, double-check that every word appears exactly once.\n\n"
        "Required JSON schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n"
    )
    return prompt

def build_student_prompt(
        words: Sequence[str],
        categories: Sequence[str],
        cfg: ConnectionsPromptConfig,
) -> str:
    return build_teacher_prompt(words, categories, cfg)

def gold_answer_json(
        categories_and_words: Sequence[Dict[str, Any]],
) -> str:
    groups = []
    for a in categories_and_words:
        cat = normalize_word(a["answerDescription"])
        ws = [normalize_word(w) for w in a["words"]]
        groups.append({"category": cat, "words": ws})

    obj = {"groups": groups}
    return json.dumps(obj, ensure_ascii=False)

def attach_reasoning_to_gold(gold_json: str, reasoning: str,
) -> str:
    obj = json.loads(gold_json)
    obj["reasoning"] = " ".join(reasoning.strip().split())
    return json.dumps(obj, ensure_ascii=False)


