from pathlib import Path
from datetime import datetime
import argparse
import json

from ocr_v2_inference import get_ocr_v2_engine


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "ocr_v2_batch"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def run_batch_ocr(image_paths):

    engine = get_ocr_v2_engine()

    results = []

    print("=" * 72)
    print("EXPANDED V2 SINHALA OCR - BATCH INFERENCE")
    print("=" * 72)

    print(f"\nCheckpoint epoch: {engine.checkpoint_epoch}")

    if engine.validation_cer is not None:
        print(f"Validation CER: {engine.validation_cer:.4f}")

    print(f"Images: {len(image_paths)}")

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        image_path = Path(image_path)

        print("\n" + "-" * 72)
        print(f"IMAGE {index}/{len(image_paths)}")
        print("-" * 72)

        print(f"Input: {image_path}")

        try:

            result = engine.predict_line(
                image_path
            )

            predicted_text = result["text"]

            print("\nPredicted Text:")
            print(predicted_text)

            results.append({
                "image_number": index,
                "sample_id": image_path.stem,
                "image_path": str(
                    image_path.resolve()
                ),
                "predicted_text": predicted_text,
                "status": "success"
            })

        except Exception as error:

            print("\nERROR:")
            print(error)

            results.append({
                "image_number": index,
                "sample_id": image_path.stem,
                "image_path": str(image_path),
                "predicted_text": "",
                "status": "failed",
                "error": str(error)
            })

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = (
        RESULTS_DIR
        / f"ocr_v2_batch_{timestamp}.json"
    )

    successful = sum(
        1
        for item in results
        if item["status"] == "success"
    )

    json_data = {
        "timestamp": datetime.now().isoformat(),
        "ocr_engine": "CRNN-CTC Expanded V2",
        "checkpoint_epoch": engine.checkpoint_epoch,
        "validation_cer": engine.validation_cer,
        "validation_wer": engine.validation_wer,
        "total_images": len(results),
        "successful_images": successful,
        "failed_images": len(results) - successful,
        "results": results
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\n" + "=" * 72)
    print("BATCH OCR COMPLETE")
    print("=" * 72)

    print(f"Total      : {len(results)}")
    print(f"Successful : {successful}")
    print(
        f"Failed     : "
        f"{len(results) - successful}"
    )

    print("\nJSON saved:")
    print(json_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "images",
        nargs="+"
    )

    args = parser.parse_args()

    run_batch_ocr(
        args.images
    )


if __name__ == "__main__":
    main()