# Evidence coverage

Verification scoring is evidence-gated: it only calculates a score when the
uploaded image matches a local reference image whose source metadata has been
independently verified.

The included records are placeholders. They have no trusted source URLs or
event metadata, so they correctly produce zero coverage.

## Add a verified reference

For each reference image, first confirm the image and facts against a trusted
publisher or fact-checker. Then register the image with its real canonical
source URL and the original event details:

### Simple method

From the project folder, run:

```powershell
.\add_verified_source.ps1
```

Enter the requested values. The helper saves the image and source metadata in
the correct evidence collection.

### Command-line method

```powershell
.\.venv311\Scripts\python.exe .\scripts\register_verified_reference.py `
  --image "C:\path\to\verified-reference.jpg" `
  --source-name "Trusted publisher" `
  --source-url "https://publisher.example/article" `
  --publication-date "2026-10-19" `
  --event-date "2026-10-18" `
  --event-name "Exact event name" `
  --location "Colombo, Sri Lanka" `
  --original-claim "The original sourced claim" `
  --filename "verified-reference.jpg"
```

The tool copies the reference image into `data/reference_images/` and writes a
validated `verified` record to `reference_metadata.json`. It refuses incomplete
records, invalid URLs/dates, duplicate names, and unsafe file names.

## When coverage increases

Coverage can increase only when all of the following are true:

1. The uploaded image is an exact or close match to a registered reference.
2. The matching record has a real HTTP(S) source URL, publisher name, event
   name/date/location, and original claim.
3. The uploaded headline provides enough explicit details for event, time,
   location, and claim checks.

Model scores from BLIP, CLIP, and authenticity classification are not verified
evidence, so they do not increase evidence coverage.
