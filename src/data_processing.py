import os 
import pandas as pd
from sklearn.model_selection import train_test_split 
import json 
import logging 
import random

#Log Config 
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_swow_data(file_path, sample_size=None, seed=42):
    '''
    Load the SWOW dataset, only the cue, R1, R2, R3 columns.
    If sample_size is provided, loads a random sample of n rows using skiprows logic.
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading dataset from {file_path}...")
    
    if sample_size is not None:
        logger.info(f"Sampling {sample_size} random rows...")
        random.seed(seed)
        
        # Conta le righe totali (escluso header)
        try:
            # Conta rapida delle linee
            with open(file_path, 'r', encoding='utf-8') as f:
                n_total_lines = sum(1 for _ in f) - 1
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin1') as f:
                n_total_lines = sum(1 for _ in f) - 1

        if sample_size < n_total_lines:

            n_skip = n_total_lines - sample_size
            skip = sorted(random.sample(range(1, n_total_lines + 1), n_skip))
        else:
            logger.warning(f"Sample size ({sample_size}) >= total rows ({n_total_lines}). Loading all data.")
            skip = None
    else:
        skip = None

    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', skiprows=skip, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, on_bad_lines='skip', skiprows=skip, encoding = 'latin1')

    #Standardization of column's names 
    df.columns = [column.lower() for column in df.columns]

    required_cols = ['cue', 'r1', 'r2', 'r3']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"The CSV must contain the following columns: {required_cols}")
    
    initial_len = len(df)
    df = df.dropna(subset=required_cols)
    logger.info(f"Loaded rows: {initial_len}. Ok rows after dropna: {len(df)}")

    return df 

def format_swow_entry(row):
    '''
    Format each row of the dataframe in the required format for the Phase 1 training of the Teacher.
    Prompt: "Word association task. Input: [cue]. Associated words:"
    Completion: " [R1], [R2], [R3]."
    '''
    cue = str(row['cue']).strip()
    r1 = str(row['r1']).strip()
    r2 = str(row['r2']).strip()
    r3 = str(row['r3']).strip()

    prompt = f"Word association task. Input: {cue}. Associated words:"
    completion = f"{r1}, {r2}, {r3}."

    return {
        "prompt": prompt,
        "completion": completion,
        "full_text": prompt + completion
    }

def process_swow(input_path, output_dir, test_size=0.05, seed=42, sample_size=None):
    '''
    This function loads, formats, splits (95% test / 5% validation) and save as a JSONL the SWOW dataset
    sample_size: Number of random examples to load. If None, loads all.
    '''
    df = load_swow_data(input_path, sample_size=sample_size, seed=seed)
    logger.info("Data format in progress...")

    formatted_data = df.apply(format_swow_entry, axis=1).tolist()
    logger.info(f"Splitting dataset (Test size: {test_size})...")

    train_data, val_data = train_test_split(formatted_data, test_size=test_size, random_state=seed)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "swow_train.jsonl")
    val_path = os.path.join(output_dir, "swow_val.jsonl")

    logger.info(f"Saving train set on {train_path} ({len(train_data)} examples)...")
    with open(train_path, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')

    logger.info(f"Saving validation set on {val_path} ({len(val_data)} examples)...")
    with open(val_path, 'w', encoding='utf-8') as f:
        for entry in val_data:
            f.write(json.dumps(entry) + '\n')
    
    logger.info("Data Processing Complete.")

if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "SWOW_EN.csv")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

    try:
        # Esempio: carica solo 50k righe per fare veloce
        process_swow(RAW_DATA_PATH, PROCESSED_DIR, sample_size=50000)
    except Exception as e:
        logger.error(f"Error during execution of Data Preprocessing: {e}")