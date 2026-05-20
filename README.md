# lcrnet-crime-classification-xai
LCRNet — Lightweight CNN + SAS-Transformer for 128-class LAPD crime classification. 99.52% test accuracy on 815K records with SHAP explainability and fairness analysis.

# 🔍 LCRNet — Crime Type Classification with XAI

Lightweight Crime Recognition Network (CNN + SAS-Transformer) for automatic classification of LAPD crime records into 128 crime categories with full SHAP explainability.

> M.Tech Final End-Semester Project · Mohit Tiwari

---

## 🏆 Results

| Metric | Score |
|---|---|
| Test Accuracy | **99.52%** |
| Weighted F1 | 0.9932 |
| Weighted Precision | 0.9917 |
| Weighted Recall | 0.9952 |
| Macro F1 | 0.6333 *(128 classes, extreme imbalance)* |
| Parameters | ~0.22M |

---

## 🧠 Architecture — LCRNet


Input (237 features)
→ InputBatchNorm          # normalizes TF-IDF + numerical + categorical to same scale
→ CNN (Conv1d ×2, GELU)   # local feature interaction extractor
→ SAS-Transformer          # global context encoder (Simulated Annealing Sparsity)
→ AdaptiveAvgPool1d        # sequence → single vector
→ FC (256 → 128 classes)   # classifier


> Based on Lu et al., 2025 — doi: [10.1038/s41598-025-07260-7](https://doi.org/10.1038/s41598-025-07260-7)

---

## 📦 Dataset

- **LAPD Crime Data 2020–Present** · 815,882 records · 28 columns
- After rare-class filter: **128 classes**, 815,879 samples
- Split: **70% train / 15% val / 15% test** (stratified)

---

## ⚙️ Feature Pipeline (No-Leakage)

| Feature Type | Columns | Preprocessing |
|---|---|---|
| TF-IDF | `Crm Cd Desc` | 231 vocab, `sublinear_tf=True`, `min_df=3` |
| Numerical | `Vict Age`, `TIME OCC` | StandardScaler (fit on train only) |
| Categorical | `Vict Sex`, `Vict Descent`, `Weapon Desc`, `AREA NAME` | LabelEncode → normalize to [0,1] |

> **Key fix:** Raw categorical codes (0–50) dominated CNN gradients. Normalizing all features to the same scale was the fix that pushed accuracy from 30% → 99.52%.

---

## 🔬 Key Technical Details

- **SAS Attention** fully implemented — `k(t) = kmax − (kmax−kmin)·(1 − e^{−β·t})`; disabled at 5 epochs (α=0.075, only 7.5% active — needs 30+ epochs per paper)
- **AMP** (`torch.cuda.amp`) for mixed-precision training on dual T4 GPUs
- **DataParallel** across 2× Tesla T4 GPUs, batch size 512
- **Label smoothing = 0.1** — prevents overconfidence on dominant classes
- **Gradient clipping** at norm=1.0 for training stability
- **No LR scheduler** — plain Adam sufficient after correct feature scaling

---

## 📊 SHAP Explainability (6 Components)

- **Global Feature Importance** — top feature: crime description token `01` (SHAP=0.095)
- **Beeswarm Summary** — per-sample SHAP distribution for dominant class
- **Class-Specific Importance** — top-5 features per top-3 crime classes
- **Waterfall Plots** — per-sample prediction explanation (3 samples)
- **Fairness / Bias Analysis** — demographic SHAP vs content SHAP
- **Dependence Plot** — feature value vs SHAP contribution

### Fairness Finding
| Attribute | Mean SHAP | vs Content Features |
|---|---|---|
| Victim Sex | 0.0002 | 387× BELOW avg |
| Victim Descent | 0.0000 | 1967× BELOW avg |

> Model predictions driven by crime description text, not demographic attributes.

---

## ⚙️ Stack

`Python` `PyTorch` `scikit-learn` `SHAP` `TF-IDF` `AMP` `DataParallel` `Seaborn`

---

## 🖥️ Environment

| | |
|---|---|
| Platform | Kaggle Notebooks |
| GPU | 2× Tesla T4 |
| PyTorch | 2.10.0+cu128 |
| CUDA | ✅ |
