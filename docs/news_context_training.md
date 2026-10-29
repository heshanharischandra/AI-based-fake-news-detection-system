# Dataset-backed news image context classifier

## Scope

The training workflow uses the official **NewsCLIPpings merged_balanced** dataset:
71,072 training pairs, 7,024 validation pairs, and 7,264 test pairs. Labels distinguish
original VisualNews image-caption pairs from constructed mismatches. They do not
label an entire article as true/false, nor an image as AI-generated/photographic.

Sources: [official dataset and feature documentation](https://github.com/g-luo/news_clippings),
[official asset repository](https://huggingface.co/g-luo/news-clippings).
Assets are pinned to revision `a2bbe9a8125f47a5c26338125c71e05be0266cf7`.
The asset repository lists Apache-2.0; this is not a blanket permission to redistribute
underlying publisher photographs or article text. This workflow downloads features
and labels, not publisher image archives or scraped news articles.

## Reproduce

From the repository root in PowerShell:

```powershell
& ./.venv311/Scripts/python.exe scripts/train_news_context.py
```

Optional individual stages: `--stage download`, `--stage prepare`, `--stage train`.
Downloads total approximately 2.2 GB. Existing validated files are reused. Repository
file sizes and available LFS SHA-256 checksums are checked. Remote feature files are
read through a restricted NumPy-only unpickler, never unrestricted `pickle.load`.
The dataset manifest records URLs, sizes, pinned revision, and local SHA-256 hashes.

## Training protocol

- The OpenAI CLIP ViT-B/32 encoder remains frozen. A regularized logistic classifier
  is trained on normalized image/text feature products, absolute differences, and
  cosine similarity. Dataset IDs, source category, and construction similarity scores
  are **not** input features.
- Official split boundaries are preserved. Repeated pairs and shared VisualNews IDs
  across splits cause preparation to fail. This ID audit cannot detect every visually
  duplicated photograph across different IDs.
- Regularization and a mismatch threshold are selected using validation balanced
  accuracy. Test data is reserved for reporting, not model selection.
- The report includes accuracy, precision, recall, F1, AUC, Brier score, confusion
  matrix, an independently validation-tuned cosine baseline, and export parity.
- Inference uses the **raw caption without prompt templates**, matching the official
  training feature recipe. Sinhala/translated headlines, screenshots, and other
  distribution shifts need separate evaluation before making accuracy claims.

## Outputs and application behavior

- Dataset/provenance: `data/training/newsclippings/manifest.json`
- Derived feature datasets: `data/training/newsclippings/{train,val,test}.npz`
- Classifier weights (plain JSON): `models/news_context.json`
- Measured results: `reports/news_context_training.json`

The demo automatically loads the trained head during CLIP analysis and shows its
image-caption mismatch score in Step 8. The evidence-based verification score stays
separate. A dataset prediction never sets evidence status to verified or fabricates
event dates, locations, source URLs, or factual support. Steps 9 and 10 remain hidden.

This is a trained context baseline, not end-to-end retraining of OCR, BLIP, CLIP, or
the AI-image detector. There is no guaranteed 100% accuracy and no automatic global
news search in this implementation.
