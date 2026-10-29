
# AI-Based Fake News Detection System

## Component: Explainable Multi-Format Fake News Detection

**Research Project:** AI-Based Fake News Detection System  
**Component Focus:** Sinhala and English Fake News Classification  
**Input Formats:** Text, Image, PDF  
**Development Status:** PP1 Prototype — Under Development

## 1. Overview

This component focuses on detecting potential fake news from Sinhala and English textual content using Machine Learning and Natural Language Processing (NLP).

The system is designed to accept three input formats: direct text, news images, and PDF documents.

Text extracted from images and PDFs is processed through a text classification pipeline to generate REAL or FAKE predictions.

The component aims to incorporate Explainable AI (XAI) to improve the interpretability of model predictions.

## 2. Research Objectives

- Develop a bilingual fake news classification system supporting Sinhala and English.
- Train and evaluate transformer-based text classification models.
- Extract textual information from news images using Optical Character Recognition (OCR).
- Support text extraction from PDF documents.
- Integrate multiple input formats into a unified classification workflow.
- Incorporate Explainable AI for interpretable predictions.
- Evaluate model performance using appropriate classification and OCR metrics.

## 3. Proposed System Architecture

**Text Input** → Text Preprocessing → Text Classification

**Image Input** → Image Preprocessing → OCR → Text Classification

**PDF Input** → PDF Text Extraction → Text Classification

**Classification Output** → REAL/FAKE Prediction → Confidence Score → Explainable AI (Planned)

All supported input formats are intended to use a common text-based classification pipeline.

This is a **multi-format input approach**, not a multimodal feature-fusion model. Image visual features are not currently combined with text features for classification.

## 4. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| NLP | Hugging Face Transformers |
| Text Classification | mBERT, XLM-RoBERTa |
| OCR | Custom CRNN with CTC, Tesseract OCR |
| Image Processing | OpenCV, Pillow |
| Development Environment | VS Code, Windows |
| Prototype UI | Streamlit (planned for integration) |
| Final Mobile Application | Group-level mobile integration |

## 5. Implemented Features

### 5.1 Sinhala and English Text Classification

- Prepared Sinhala and English news datasets.
- Performed text cleaning and preprocessing.
- Prepared binary REAL and FAKE classification labels.
- Trained and evaluated multilingual transformer models.
- Compared mBERT and XLM-RoBERTa classification approaches.
- Implemented text classification inference scripts.
- Investigated class imbalance affecting Sinhala fake news classification.

### 5.2 Sinhala OCR Model Development

Developed a custom Sinhala Optical Character Recognition model using a Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC).

Development included:

- Synthetic Sinhala text dataset generation.
- Real Sinhala news image collection.
- Manual text-line annotation.
- Image preprocessing and segmentation.
- Data augmentation.
- CRNN training and fine-tuning.
- CER and WER evaluation.
- Multiple OCR model improvements through V2, V3, V4 and V5 experiments.

### 5.3 Multilingual News Image OCR

Implemented a development-stage multilingual image text extraction pipeline featuring:

- News image loading and preprocessing.
- Automatic headline-region detection.
- Text-line segmentation and cropping.
- Sinhala and English language handling.
- Tesseract-based English OCR.
- Custom CRNN-based Sinhala OCR.
- Hybrid Sinhala OCR experiments.
- Extracted text and diagnostic output generation.

The V4.6 hybrid image OCR pipeline is available in the source code. Integration of the newly trained V5 recognizer remains in progress.

## 6. Sinhala OCR V5 Training Results

The V5 model was fine-tuned using real Sinhala news text-line images and augmented training samples.

### Dataset Summary

| Dataset Property | Count |
|---|---:|
| Original real training lines | 121 |
| Augmented training lines | 1,452 |
| Total training samples | 1,573 |
| Validation samples | 23 |
| Training sources | 35 |
| Validation sources | 6 |
| Training-validation source leakage | 0 |

### Model Evaluation

| Evaluation Metric | V3 Baseline | V5 Model |
|---|---:|---:|
| Character Error Rate (CER) | 0.4055 | 0.3470 |
| Word Error Rate (WER) | 0.9167 | 0.8214 |

**Best V5 Epoch:** 13

The V5 model achieved approximately a 14.4% relative reduction in CER and a 10.4% relative reduction in WER on the development-validation dataset.

These results are based on a small, source-isolated validation set and should not be interpreted as final generalization performance. Final unseen-image evaluation is still pending.

## 7. Important Source Files

| File | Purpose |
|---|---|
| `src/predict.py` | Text classification inference |
| `src/predict_sinhala_v3.py` | Sinhala classification inference |
| `src/image_ocr.py` | Initial image text extraction |
| `src/image_ocr_v4_hybrid.py` | Image headline detection and extraction |
| `src/image_ocr_v4_5_multilingual.py` | Multilingual image OCR |
| `src/image_ocr_v4_6_hybrid_multilingual.py` | Hybrid multilingual image OCR |
| `src/ocr_crnn_model.py` | Custom CRNN model architecture |
| `src/ocr_dataset.py` | OCR dataset processing |
| `src/ocr_v3_inference.py` | V3 Sinhala OCR inference |
| `src/train_ocr_crnn.py` | Initial CRNN model training |
| `src/finetune_real_ocr_v5.py` | V5 Sinhala OCR fine-tuning |
| `src/evaluate_ocr_crnn.py` | OCR model evaluation |

Additional preprocessing, augmentation, validation, training and experimental scripts are available under `src/`.

## 8. Development Setup

### Prerequisites

- Python 3.11
- Git
- A Python virtual environment
- Required Python packages
- Tesseract OCR with Sinhala and English language data
- Appropriate trained model checkpoints and dataset files

### Create a Virtual Environment

From the repository root:

`py -3.11 -m venv .venv`

Activate it in Windows PowerShell:

`.\.venv\Scripts\Activate.ps1`

Required dependencies include PyTorch, Transformers, OpenCV, Pillow, NumPy, Pandas, scikit-learn and pytesseract.

A complete pinned dependency file is still to be prepared.

### Run Text Classification

`python src/predict_sinhala_v3.py`

### Run Multilingual Image OCR

`python src/image_ocr_v4_6_hybrid_multilingual.py`

**Important:** The scripts require additional locally stored model checkpoints, data files and environment configuration. Some development scripts also contain machine-specific paths. The repository is therefore not yet a fully reproducible, ready-to-run release.

## 9. Dataset and Model Availability

Large datasets, trained model checkpoints, generated results, virtual environments and private configuration files are excluded from Git using `.gitignore`.

This repository primarily contains source code and development scripts.

Trained model artifacts and required datasets must be obtained or prepared separately before model inference and training can be reproduced.

## 10. Current Limitations

- Sinhala fake news classification remains affected by dataset class imbalance.
- High overall text-classification accuracy does not guarantee strong minority-class detection.
- Sinhala OCR performance is weaker on challenging real-world news images.
- The V5 OCR recognizer has not yet been fully integrated into the image extraction pipeline.
- The complete Text–Image–PDF workflow is not yet integrated.
- Explainable AI functionality remains under development.
- Final unseen-data evaluation and complete end-to-end testing are pending.

## 11. Future Development

- Complete V5 OCR inference integration.
- Improve Sinhala OCR recognition on unseen news images.
- Integrate OCR output with the text classifier.
- Implement and connect PDF text extraction.
- Develop a temporary Streamlit interface for Text, Image and PDF inputs.
- Reject unsupported file formats.
- Implement Explainable AI methods.
- Improve Sinhala FAKE-class classification performance.
- Conduct final evaluation and prepare deployment documentation.
- Integrate the component with the group's final application.

## 12. Research Note

Model predictions represent classification estimates based on the available training data. A REAL prediction or high confidence score does not independently verify that a news claim is factually true.

The system is intended to support fake news detection research and assist further verification.

## 13. Development Status

**PP1 Progress:** Working text classification and Sinhala/English OCR research prototypes, with model training and evaluation completed.

**Current Focus:** OCR V5 integration, unified input processing, temporary frontend development, and explainability.

---

**Project:** AI-Based Fake News Detection System  
**Component:** Explainable Multi-Format Fake News Detection  
**Supported Languages:** Sinhala and English