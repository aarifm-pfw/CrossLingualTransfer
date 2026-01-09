# NLP_project

A small collection of experiments and scripts for natural language processing tasks, primarily exploring:
- Zero-shot classification using Hugging Face transformers (BART MNLI).
- Abstractive summarization training / evaluation (BART).
- Cross-lingual evaluation and transfer experiments.

This repository contains a Colab / Jupyter notebook with the main project flow and two Python scripts for smaller experiments and utilities.

---

## Repository contents

- `project_code.ipynb` — Primary Jupyter/Colab notebook. Contains end-to-end experiments: dataset loading (e.g., XLSum English subset), model setup (BART), training with `transformers.Trainer`, evaluation, logging (Weights & Biases), and some zero-shot classification usage. Intended for interactive reproduction in Colab or a local Jupyter environment.
- `cross-lingual-transfer.py` — Script with a `CrossLingualZeroShot` class (and `main()` entry). Implements cross-lingual zero-shot classification workflows, evaluation (accuracy, F1), and BLEU scoring for translations where applicable.
- `project_iteration0.py` — Lightweight script demonstrating zero-shot classification (facebook/bart-large-mnli) and evaluating on a sample dataset (uses `ag_news` as an example).
- `README.md` — This file.

---

## Key functionality / highlights

- Zero-shot classification using `facebook/bart-large-mnli` via the transformers `pipeline("zero-shot-classification")`.
- Summarization experiments using `facebook/bart-large-cnn` (or similar BART variants) and the `Trainer` API for fine-tuning on datasets like `csebuetnlp/xlsum`.
- Dataset loading via the `datasets` library (`load_dataset`).
- Evaluation metrics include classification accuracy, precision/recall/F1 (scikit-learn), and BLEU (NLTK smoothing).
- Optional logging through Weights & Biases (wandb) as seen in the notebook.

---

## Requirements

Recommended Python version: 3.8 — 3.11

Core Python packages used across the notebook and scripts:
- transformers
- datasets
- torch
- numpy
- pandas
- scikit-learn
- nltk
- tqdm
- wandb (optional — used in notebook logging)
- sentencepiece (sometimes required by tokenizers)
- (Optional) colab libraries if running in Google Colab

Example pip install (adjust versions as needed):
```bash
python -m pip install --upgrade pip
pip install transformers datasets torch numpy pandas scikit-learn nltk tqdm wandb sentencepiece
```

If you prefer a pinned requirements file, you can create one from the environment you test in and add it to the repo.

---

## Setup

1. (Optional but recommended) Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows (PowerShell)
```

2. Install dependencies:
```bash
pip install transformers datasets torch numpy pandas scikit-learn nltk tqdm wandb sentencepiece
```

3. (Optional) Download NLTK BLEU smoothing tools (if running BLEU evaluation through NLTK):
```python
python -c "import nltk; nltk.download('punkt')"
```

4. (Optional) Configure tokens:
- Hugging Face Hub: set `HF_TOKEN` as an environment variable or use the local CLI `huggingface-cli login`.
- Weights & Biases (if using wandb logging in the notebook): set `WANDB_API_KEY` or log in interactively.

---

## Quick start — Running the scripts

Notes:
- Large models (BART variants) are resource-intensive. Use a machine with a GPU for training or heavy inference.
- The notebook (`project_code.ipynb`) was authored to run in Google Colab (GPU) — it contains `!pip install` cells and example setup.

A. Run the zero-shot demo script:
```bash
python project_iteration0.py
```
This runs a few example zero-shot classification inferences and shows how to evaluate on an example dataset (AG News).

B. Run cross-lingual zero-shot evaluation:
```bash
python cross-lingual-transfer.py
```
(If the script accepts CLI args in your local copy, pass dataset/model paths or flags accordingly. See the top of the script or add `--help` if implemented.)

C. Run the notebook:
- Open `project_code.ipynb` in Jupyter or Colab.
- Follow the cells: install required packages, load datasets, authenticate to Hugging Face if needed, and run training/evaluation cells.
- If running in Colab, enable GPU (Runtime > Change runtime type > GPU) for training steps.

---

## Datasets referenced

- `csebuetnlp/xlsum` (English subset used for summarization experiments)
  - Used in the notebook for fine-tuning/evaluating BART summarizers.
- `ag_news`
  - Used in `project_iteration0.py` as a sample classification dataset for zero-shot evaluation.

If you run experiments locally, the `datasets` library will download these automatically when `load_dataset(...)` is called.

---

## Models referenced

- `facebook/bart-large-mnli` — used for zero-shot classification (via `pipeline("zero-shot-classification")`).
- `facebook/bart-large-cnn` (or other BART generation models) — used for summarization fine-tuning and generation in the notebook.

You can replace these with other Hugging Face models as long as they match the expected task (i.e., a sequence classification-capable checkpoint for zero-shot, and an encoder-decoder generation model for summarization).

---

## Evaluation & Metrics

- Classification:
  - Accuracy, Precision, Recall, F1 (scikit-learn `classification_report`, `accuracy_score`, `f1_score`).
- Summarization:
  - BLEU (NLTK) is used in cross-lingual/blind translation checks in `cross-lingual-transfer.py`.
  - For summarization, consider using ROUGE or BERTScore (not currently in the repo) for more appropriate evaluations.

---

## Reproducing experiments (suggested steps)

1. Get a machine with a GPU and sufficient RAM/disk for datasets + models.
2. Prepare environment and install dependencies (see Setup).
3. If using the notebook:
   - Run the install cells to ensure dependencies are present.
   - Set HF and W&B tokens if you want to push logs or use private models/datasets.
   - Run dataset loading cell (e.g., `dataset = load_dataset("csebuetnlp/xlsum", "english")`).
   - Initialize tokenizer and model (e.g., `BartTokenizer.from_pretrained("facebook/bart-large-cnn")`).
   - Configure `TrainingArguments` appropriately (batch size, learning rate, number of epochs) and run trainer.
4. If running `cross-lingual-transfer.py`, inspect the script header to find expected inputs / edit defaults to point to your dataset and model choices.

---

## Notes, caveats and tips

- Model sizes:
  - `facebook/bart-large-*` models are large (~1.6B parameters for larger variants). Fine-tuning on CPU is impractical; use GPU or smaller models for experimentation.
- Runtime / memory:
  - Use smaller batch sizes if you encounter OOM errors. Consider gradient accumulation to emulate larger batch sizes.
- Tokenizer differences:
  - Ensure tokenizer and model come from the same family (e.g., BART tokenizer for BART model).
- Logging:
  - The notebook contains hooks to log to Weights & Biases. If you don't have a WANDB account or don't want to log, disable or remove the wandb.init() / Trainer callbacks that use wandb.

---

## Contributing

This repository appears to be a personal project or experiment repository. Suggestions to improve:
- Add a `requirements.txt` or `environment.yml` with pinned versions for reproducibility.
- Add a LICENSE file (e.g., MIT) if you want to permit reuse.
- Add a lightweight CLI or argument parsing to `cross-lingual-transfer.py` for configurable dataset/model/paths.
- Extract common utilities (data preprocessing, metric functions) into a `src/` package for cleaner reuse.

If you'd like, I can:
- Create a suggested `requirements.txt`.
- Propose a sample CLI wrapper for `cross-lingual-transfer.py`.
- Add a minimal `setup.py` / `pyproject.toml` for packaging.

---

## License

No license file is present in the repository. If you want this project to be reusable by others, add a license (for example, MIT). If you'd like, I can prepare a LICENSE file for you.

---

## Contact / Author

Repository owner: aarifm-pfw

---

If you want, I can:
- Produce a ready-to-add `requirements.txt`.
- Add a `LICENSE` (MIT template).
- Generate a small example script that downloads a subset of XLSum and runs a short training loop (fast debug mode) for CI-friendly checks.

Tell me which of these you'd like next and I will prepare the files.
``` ````