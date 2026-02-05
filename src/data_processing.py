import os 
import pandas as pd
from sklearn.model_selection import train_test_split 
import json 
import logging 

#Log Config 
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_swow_data(file_path):
    '''
    Load the SWOW dataset, only the cue, R1, R2, R£ columns 
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading dataset from {file_path}...")

    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, on_bad_lines='skip', encoding = 'latin1')

    #Standardization of column's names 
    df.columns = [column.lower() for column in df.columns]

    required_cols = ['cue', 'r1', 'r2', 'r3']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"The CSV must contain the following columns: {required_cols}")
    
    initial_len = len(df)
    #initial_len 
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

def process_swow(input_path, output_dir, test_size=0.05, seed=42):
    '''
    This function loads, formats, splits (95% test / 5% validation) and save as a JSONL the SWOW dataset
    '''
    df = load_swow_data(input_path)
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
        process_swow(RAW_DATA_PATH, PROCESSED_DIR)
    except Exception as e:
        logger.error(f"Error during execution of Data Preprocessing: {e}") 




