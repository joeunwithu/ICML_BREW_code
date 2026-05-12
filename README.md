# ICML_BREW

This repository provides a lightweight reference implementation of BREW for understanding the main algorithmic components.
The released code is intended as an illustrative implementation rather than an exact reproduction package for all experiments reported in the paper. For simplicity and readability, some engineering details are simplified.

## 1. Environment

```bash
conda create -n BREW python=3.10 -y
conda activate BREW
conda install -c conda-forge sage -y
pip install -r requirements.txt
```

`SageMath` is required because the watermark code imports `sage.all`. Install it with conda, not pip.

For a CUDA-specific PyTorch build, install PyTorch with the command matched to the server CUDA version from the official PyTorch selector, then run `pip install -r requirements.txt` for the remaining packages.

## 2. Model

The default model path is:

```text
models/facebook/opt-1.3b
```

The first run downloads `facebook/opt-1.3b` automatically if `models/facebook/opt-1.3b/config.json` does not exist. For private or rate-limited Hugging Face access, set a token first.

```bash
export HF_TOKEN=your_huggingface_token
```

## 3. Run

Run BREW on the first C4 and OpenGen samples.

```bash
python run.py
```

Run BREW explicitly.

```bash
python run.py --algorithms BREW
```

Run C4 only.

```bash
python run.py --dataset c4 --max-samples 1
```

Run OpenGen only.

```bash
python run.py --dataset opengen --max-samples 1
```

Run more samples.

```bash
python run.py --dataset all --max-samples 10
```

Use an existing local model path.

```bash
python run.py --model-path /path/to/opt-1.3b
```

## 4. Outputs

Results are saved under:

```text
results/BREW/
```

Each sample directory contains watermarked, unwatermarked, and natural-text detection outputs. `dataset/c4/processed_c4.json` and `dataset/openGen/OpenGen.jsonl` are used by `run.py`. The algorithm directory also contains `summary.json` and `time_taken.txt`.

## 5. Algorithm name

The available algorithm name is `BREW`.
