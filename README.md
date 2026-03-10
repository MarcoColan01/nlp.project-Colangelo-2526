# No Spoiler Please! A TinyBERT Model to Detect Spoiler Reviews on IMDb
This repository contains the project for the Natural Language Processing course of the master degree in computer science at Università degli Studi di Milano. The project has been developed by Marco Colangelo (ID 67045A).

This project focus on **knowledge distillation** applied to **spoiler detection in IMDb movie reviews**. The goal is to transfer the predictive behaviour of a fine tuned **BERT-base** teacher model to a smaller **TinyBERT_General_4L_312D** student model, obtaining a substantially lighter model with limited performance degradation.

The project follows a modular design:
- Reusable logic is implemented in `src/`
- The end-to-end workflow is executed through Jupyter notebooks
- Intermediate datasets, checkpoints and plots are stored locally

The overall pipeline is:
1. Download the raw dataset from Kaggle
2. Preprocess reviews with an head+tail truncation strategy
3. Fine tune of the BERT teacher model
4. Distillation of the TinyBERT model in two distinct phases
5. Generate evaluation and interpretation plots

## Project Overview
The task is a binary classification problem: given an IMDb review, predict whether it contains spoilers.

### Models
- **Teacher model:** [`bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased)
    - Layers: 12
    - Attention heads: 12
    - Hidden states: 768
    - Embeddings: 768
    - Total parameters: 110M
- **Student model:** [`TinyBERT_General_4L_312D`](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)
    - Layers: 4
    - Attention heads: 12
    - Hidden states: 312
    - Embeddings: 312
    - Total parameters: 14.5M

### Training Strategy
- **Teacher fine tuning:** 
    - Weighted cross-entropy to mitigate class imbalance
    - Head+tail truncation strategy to overcome the maximum 512 token limit in BERT
    - Layer-wise learning rate decay to avoid catastrophic forgetting
- **Student distillation phase 1:** embedding, hidden-states and attention alignment
- **Student distillation phase 2:** prediction-layer distillation using teacher logits and hard labels

### Main Objective
The final distilled student is intended to preserve most of the teacher's performance while significantly reducing model size and inference time.

## Repository Structure
```text
.
├── src/
│   ├── kaggle_download.py
│   ├── imdb_spoiler_io.py
│   ├── splitters.py
│   ├── head_tail.py
│   ├── teacher_finetune_headtail.py
│   ├── distillation.py
│   └── project_paths.py
├── notebooks/
│   ├── 1_headtail.ipynb
│   ├── 2_teacher_finetune.ipynb
│   ├── 3_student_distillation_phase1.ipynb
│   ├── 4_student_distillation_phase2.ipynb
│   ├── 5_evaluation_graphs.ipynb
│   └── 6_emb_attn_exploration.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── checkpoints/
├── requirements.txt
└── README.md
```
## Getting Started 
First of all, clone the repository with the following commands.
```bash
git clone https://github.com/MarcoColan01/nlp.project-Colangelo-2526.git
cd nlp.project-Colangelo-2526
```
### Environment Setup
It is recommended to use a virtual environment
```bash
python -m venv .venv
#source .venv/bin/activate  #For Linux/macOS
#.venv\Scripts\activate     #For Windows

pip install -r requirements.txt
```
### Recommended requirements (according to the specifications of the desktop PC used for the project)
- Python 3.10
- 32GB of RAM
- Nvidia GeForce RTX 3070 (8GB of VRAM) or superior

In general, a GPU is highly recommended.

## Dataset Download
The project uses the [**IMDb Spoiler Dataset**](https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset?select=IMDB_reviews.json) from Kaggle.
To download it, you need a valid Kaggle API token (`kaggle.json`).

Supported options: 
- Place `kaggle.json` in the default Kaggle location: `~/.kaggle/kaggle.json`
    ```bash
    python src/kaggle_download.py --dataset-slug OWNER/DATASET_NAME
    ```
- Pass its path explicitly at runtime
   ```bash
   python src/kaggle_download.py --dataset-slug OWNER/DATASET_NAME --kaggle-json /path/to/kaggle.json
    ```

## Data Preprocessing
The preprocessing logic is split across: 
- `imdb_spoiler_io.py` for loading and normalizing the raw dataset
- `splitters.py` for grouped train/validation/test splitting by `movie_id`
- `head_tail.py` for head+tail truncation and token preparation

The split is performed by grouping samples according to the `movie_id` attribute, so that reviews of the same movie do not appear in different splits. 

The preprocessing notebook produces `.parquet` files for train, validation and test.

## Execution Order
Run the modules and notebooks in the following order.

### 1. `notebooks/1_headtail.ipynb`
This notebook:
- loads the raw IMDb spoiler data
- samples the working subset
- performs the grouped train/validation/test split
- applies head+tail truncation
- saves the processed parquet files by the rest of the pipeline

### 2. `notebooks/2_teacher_finetune.ipynb`
This notebook:
- loads the processed datasets
- fine tunes the BERT teacher model on spoiler detection
- evaluates the resulting teacher 
- saves the resulting checkpoint(s)

### 3. `notebooks/3_student_distillation_phase1.ipynb`
This notebook:
- loads the fine tuned teacher
- initializes the TinyBERT student
- performs intermediate-layer distillation
- aligns embeddings, hidden states and attention heads
- saves phase-1 student checkpoints

### 4. `notebooks/4_student_distillation_phase2.ipynb`
This notebook:
- loads the phase 1 student and the fine tuned teacher
- performs prediction-layer distillation
- combines soft supervision from teacher logits with hard-label supervision 
- saves the final distilled student model

### 5. `notebooks/5_evaluation_graphs.ipynb`
This notebook generates the main performance and efficiency plots, including:
- confusion matrices
- grouped metric bar charts
- performance vs inference-time scatter plots
- performance vs model-size scatter plots

### 6. `notebooks/6_emb_attn.ipynb`
This notebook produces qualitative interpretability plots, including:
- PCA projections of teacher and student representations
- attention heatmaps on selected challenging examples

## Expected Outputs

After running the full pipeline, the repository should contain:
- raw data in `data/raw/`
- processed parquet files in `data/processed/`

- teacher checkpoints in `checkpoints/`

- student phase-1 checkpoints in `checkpoints/`

- student phase-2 checkpoints in `checkpoints/`

- evaluation and interpretability plots generated by the last notebooks

## Reproducibility Notes

To reproduce the results:

1. run the notebooks in the exact order reported above

2. keep the same processed splits across all stages

3. use the same teacher checkpoint for both distillation phases

4. use the same phase-1 checkpoint as input to phase 2

5. keep package versions fixed through `requirements.txt`

## Results Summary

The distilled student remains close to the fine-tuned teacher while offering substantial efficiency gains. In the final report, the student shows:

- only a limited drop in PR-AUC with respect to the fine-tuned teacher

- a large reduction in the number of parameters

- a large reduction in inference time

This makes the distilled model the best efficiency--performance compromise among the compared models.

## Report and slides

The full project [report](report/sn-article.pdf) is available in this repository, as weel as the presentation [slides]().

## References
- Knowledge Distillation and TinyBERT structure. The first paper introduces the concept of Knowledge Distillation, the second paper introduces the TinyBERT model and the two phase task-specific distillation procotol and the third paper introduces the concept of "imperfect teacher", useful to correct prediction-layer distillation including also hard labels.
    - Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
    - Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., ... & Liu, Q. (2020, November). Tinybert: Distilling bert for natural language understanding. In Findings of the association for computational linguistics: EMNLP 2020 (pp. 4163-4174).
    - Ji, G., & Zhu, Z. (2020). Knowledge distillation in wide neural networks: Risk bound, data efficiency and imperfect teacher. Advances in Neural Information Processing Systems, 33, 20823-20833.
- Fine Tuning BERT for Spoiler Detection. This paper introduces a head+tail truncation strategy of IMDb reviews. Since BERT has a maximum token size of 512, review text often exceeds this limit. The authors found that, in case of overflow, the best truncation strategy (post-tokenization of reviews text) involves retaining only the first 128 tokens (useful because they introduce important context for the review) and the last 382 tokens (which often contain the actual spoiler). The special token [CLS] is attached at the beginning of the resulting sequence, while the special token [SEP] is attached at the end. Furthermore, they found that a layer-wise learning rate decay is able to avoid catastrophic forgetting during fine-tuning.
    - Sun, Qiu, Xu & Huang (2020). How to Fine-Tune BERT for Text Classification?