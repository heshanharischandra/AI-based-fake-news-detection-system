# System evaluation

Run the evaluator from the project root. It uses the original files in [uploads/](../uploads/) by default when they exist, automatically excluding the derived `uploads/processed/` and `uploads/ui/` images. Use [test_images/](../test_images/) explicitly when you have a separate held-out test set:

```powershell
.\.venv311\Scripts\python.exe .\evaluate_system.py
```

The evaluator writes a timestamped JSON report to [reports/](../reports/). It runs the deployed pipeline in this order: OCR, translation, CLIP image-text similarity, AI-authenticity detection, then local image-reuse detection.

## Ground-truth format

Create [ground_truth.json](../uploads/ground_truth.json) beside the selected image directory. The image path is always relative to that selected directory and must not leave it.

If the manifest does not exist, the evaluator automatically creates a label-free template and stops safely. Alternatively, create it explicitly after copying your images:

```powershell
.\.venv311\Scripts\python.exe .\evaluate_system.py --create-ground-truth-template
```

Complete the labels before running the evaluation. Labels must come from an independent human annotation or verified source, never from the same system output being measured.

Check which labels are missing without loading the models:

```powershell
.\.venv311\Scripts\python.exe .\evaluate_system.py --check-labels
```

Records containing only `id` and `image` provide inputs, not correct answers. A normal run now stops before model loading if no metrics have labels. To deliberately run predictions only, add `--allow-unlabelled`; accuracy values will remain `null`, with `labels_required` as the metric status. Partial annotation is supported: only annotated tasks can be scored.

For each image, manually transcribe the main headline into `ocr_text` and set `language` to `si`, `en`, or `si+en`. For CLIP, provide `comparison_text` and independently judge whether that text matches the image in `clip_match_label`. For reuse, compare against the actual local reference collection before setting `reuse_label`. Only set `authenticity_label` when the image's origin is known; omit unknown labels. Do not set them from classifier scores or filenames.

Reports separate `label_coverage` (annotations available) from `prediction_coverage_on_labelled_samples` (predictions available). Classification metrics are conditional on available predictions; check both coverage and metrics when comparing models.

```json
{
  "samples": [
    {
      "id": "sinhala-001",
      "image": "sinhala-001.png",
      "language": "si",
      "ocr_text": "නිවැරදි සිංහල ප්‍රධාන සිරස්තලය",
      "comparison_text": "A precise English description of the image",
      "clip_match_label": true,
      "reuse_label": false,
      "reuse_expected_reference": null,
      "authenticity_label": "real"
    },
    {
      "id": "english-001",
      "image": "english-001.png",
      "language": "en",
      "ocr_text": "Exact English headline",
      "comparison_text": "Exact English description used for the CLIP comparison",
      "clip_match_label": false,
      "reuse_label": true,
      "reuse_expected_reference": "known-reference.jpg",
      "authenticity_label": "ai_generated"
    }
  ]
}
```

The required labels for each metric are:

| Metric | Required fields |
| --- | --- |
| Text classification F1 | `language` |
| OCR CER | `ocr_text` |
| CLIP threshold performance | `comparison_text`, `clip_match_label` |
| Image reuse retrieval | `reuse_label`; add `reuse_expected_reference` for top-1 reference accuracy |
| AI-authenticity detection | `authenticity_label` |

`language` may be `si`, `en`, `si+en`, or `unknown`. Text-classification F1 means macro F1 for the OCR language detector because the application does not include an independent fake/real text classifier. Authenticity and CLIP scores are not truth probabilities. Use a validation split to select a CLIP threshold, then an untouched test split to report final performance.

For a different location or a fixed report name:

```powershell
.\.venv311\Scripts\python.exe .\evaluate_system.py `
  --test-dir .\test_images `
  --ground-truth .\test_images\ground_truth.json `
  --clip-threshold 0.30 `
  --report-name heldout_eval.json
```
