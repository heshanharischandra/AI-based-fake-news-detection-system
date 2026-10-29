
from hashlib import sha256
from pathlib import Path

import imagehash
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parents[2]
REFERENCE_DIR = BASE_DIR / "data" / "reference_images"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

HASH_NAMES = (
    "phash",
    "dhash",
    "whash",
    "average_hash",
    "colorhash",
)

HASH_WEIGHTS = {
    "phash": 0.40,
    "dhash": 0.25,
    "whash": 0.20,
    "average_hash": 0.10,
    "colorhash": 0.05,
}

HASH_MATCH_LIMITS = {
    "phash": 0.12,
    "dhash": 0.15,
    "whash": 0.15,
    "average_hash": 0.15,
    "colorhash": 0.20,
}


def _canonical_image(image_path):
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return image.copy()


def _file_digest(image_path):
    digest = sha256()
    with open(image_path, "rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_digest(image):
    # Exact means full-resolution pixel equality, including image geometry.
    # Resized thumbnails can hide edits and even collapse different sizes.
    digest = sha256(f"{image.mode}:{image.width}:{image.height}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def calculate_image_hashes(image_path):
    """Return several complementary perceptual hashes for an image."""
    image = _canonical_image(image_path)
    return {
        "phash": imagehash.phash(image, hash_size=16),
        "dhash": imagehash.dhash(image, hash_size=16),
        "whash": imagehash.whash(image, hash_size=16),
        "average_hash": imagehash.average_hash(image, hash_size=16),
        "colorhash": imagehash.colorhash(image),
    }


def calculate_image_hash(image_path):
    """Backward-compatible pHash helper used by older callers."""
    return calculate_image_hashes(image_path)["phash"]


def _hash_comparison(uploaded_hashes, reference_hashes):
    distances = {}
    normalized = {}
    votes = 0

    for name in HASH_NAMES:
        distance = int(uploaded_hashes[name] - reference_hashes[name])
        bit_count = max(len(uploaded_hashes[name]), 1)
        normalized_distance = distance / bit_count
        distances[name] = distance
        normalized[name] = round(normalized_distance, 4)
        if normalized_distance <= HASH_MATCH_LIMITS[name]:
            votes += 1

    weighted_score = sum(
        normalized[name] * HASH_WEIGHTS[name]
        for name in HASH_NAMES
    )

    return {
        "hash_distances": distances,
        "normalized_hash_distances": normalized,
        "hash_similarity_score": round(1.0 - weighted_score, 4),
        "weighted_hash_distance": round(weighted_score, 4),
        "hash_match_votes": votes,
    }


def _feature_match(uploaded_path, reference_path):
    """Use SIFT plus a homography to verify local visual features."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {
            "feature_status": "opencv_unavailable",
            "feature_good_matches": 0,
            "feature_inliers": 0,
            "feature_inlier_ratio": 0.0,
        }

    try:
        # Use the same EXIF-corrected pixels as the perceptual-hash path.
        uploaded = np.asarray(_canonical_image(uploaded_path).convert("L"))
        reference = np.asarray(_canonical_image(reference_path).convert("L"))
    except Exception:
        return {
            "feature_status": "feature_image_unreadable",
            "feature_good_matches": 0,
            "feature_inliers": 0,
            "feature_inlier_ratio": 0.0,
        }

    try:
        detector = cv2.SIFT_create(nfeatures=1600)
    except AttributeError:
        return {
            "feature_status": "sift_unavailable",
            "feature_good_matches": 0,
            "feature_inliers": 0,
            "feature_inlier_ratio": 0.0,
        }

    keypoints_a, descriptors_a = detector.detectAndCompute(uploaded, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(reference, None)
    if descriptors_a is None or descriptors_b is None:
        return {
            "feature_status": "insufficient_features",
            "feature_good_matches": 0,
            "feature_inliers": 0,
            "feature_inlier_ratio": 0.0,
        }

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches = [
        first
        for pair in raw_matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]
    # Several query features must not count the same reference feature as
    # independent evidence (common with repetitive patterns or lettering).
    unique_matches = {}
    for match in sorted(good_matches, key=lambda item: item.distance):
        unique_matches.setdefault(match.trainIdx, match)
    good_matches = list(unique_matches.values())

    inliers = 0
    if len(good_matches) >= 4:
        points_a = np.float32(
            [keypoints_a[item.queryIdx].pt for item in good_matches]
        ).reshape(-1, 1, 2)
        points_b = np.float32(
            [keypoints_b[item.trainIdx].pt for item in good_matches]
        ).reshape(-1, 1, 2)
        try:
            homography, mask = cv2.findHomography(
                points_a,
                points_b,
                cv2.RANSAC,
                5.0,
            )
            if homography is not None and np.isfinite(homography).all() and mask is not None:
                inliers = int(mask.ravel().sum())
        except cv2.error:
            inliers = 0

    ratio = inliers / max(len(good_matches), 1)
    return {
        "feature_status": "feature_match_calculated",
        "feature_good_matches": len(good_matches),
        "feature_inliers": inliers,
        "feature_inlier_ratio": round(ratio, 4),
    }


def _empty_result(status):
    return {
        "possible_reuse": None,
        "status": status,
        "matched_reference": None,
        "nearest_reference_candidate": None,
        "reference_count": 0,
        "hash_distance": None,
        "hash_distances": {},
        "match_confidence": None,
        "match_strength": "unavailable",
        "calibration_status": "not_calibrated_without_labeled_pairs",
        "method": "exact_sha256+perceptual_hash_ensemble+sift_confirmation",
    }


def _match_decision(comparison):
    """Use the same evidence requirements for candidate ranking and decisions."""
    if comparison["file_digest_match"] or comparison["pixel_digest_match"]:
        return (0, "exact_duplicate", True, "exact")
    feature_match = (
        comparison["feature_inliers"] >= 12
        and comparison["feature_inlier_ratio"] >= 0.25
    )
    strong_hash = comparison["hash_match_votes"] >= 3 and comparison["weighted_hash_distance"] <= 0.12
    possible_hash = comparison["hash_match_votes"] >= 3 and comparison["weighted_hash_distance"] <= 0.20
    if feature_match and strong_hash:
        return (1, "strong_near_duplicate", True, "strong_heuristic_match")
    if feature_match:
        return (2, "probable_near_duplicate", True, "probable_heuristic_match")
    if possible_hash:
        # Similar hashes alone also occur for unrelated flat/low-detail images.
        return (3, "ambiguous_near_duplicate", None, "ambiguous_requires_review")
    return (4, "no_close_local_match", False, "no_match_in_local_collection")


def check_image_reuse(image_path, threshold=6):
    """Compare an image with the local reference collection.

    ``threshold`` is retained for API compatibility only; it does not control
    the multi-signal decision and must not be displayed as its cutoff.
    """
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not REFERENCE_DIR.exists():
        return _empty_result("reference_collection_not_available")

    reference_files = sorted(
        path
        for path in REFERENCE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not reference_files:
        return _empty_result("reference_collection_empty")

    uploaded_hashes = calculate_image_hashes(image_path)
    uploaded_file_digest = _file_digest(image_path)
    uploaded_pixel_digest = _pixel_digest(_canonical_image(image_path))
    best = None
    compared_count = 0
    failed_references = []

    for reference_path in reference_files:
        try:
            reference_hashes = calculate_image_hashes(reference_path)
            comparison = _hash_comparison(uploaded_hashes, reference_hashes)
            comparison["file_digest_match"] = (
                uploaded_file_digest == _file_digest(reference_path)
            )
            comparison["pixel_digest_match"] = (
                uploaded_pixel_digest
                == _pixel_digest(_canonical_image(reference_path))
            )

            feature = {
                "feature_status": "not_needed_exact_match",
                "feature_good_matches": 0,
                "feature_inliers": 0,
                "feature_inlier_ratio": 0.0,
            }
            if not (comparison["file_digest_match"] or comparison["pixel_digest_match"]):
                try:
                    feature = _feature_match(image_path, reference_path)
                except Exception as error:
                    feature.update(feature_status="feature_comparison_failed", feature_error=str(error))
            comparison.update(feature)
            comparison["reference_path"] = reference_path
            comparison["decision"] = _match_decision(comparison)
            comparison["ranking_key"] = (
                comparison["decision"][0],
                comparison["weighted_hash_distance"],
                -comparison["feature_inlier_ratio"],
            )
            compared_count += 1
            if comparison["feature_status"] in {
                "feature_comparison_failed", "feature_image_unreadable",
                "opencv_unavailable", "sift_unavailable",
            }:
                failed_references.append({"reference": reference_path.name, "reason": comparison["feature_status"]})
            if best is None or comparison["ranking_key"] < best["ranking_key"]:
                best = comparison
        except Exception as error:
            failed_references.append({"reference": reference_path.name, "reason": str(error)})

    if best is None:
        result = _empty_result("reference_comparison_failed")
        result.update(reference_collection_count=len(reference_files),
                      failed_references=failed_references)
        return result

    _, status, possible_reuse, match_strength = best["decision"]
    if possible_reuse is False and failed_references:
        status = "reference_comparison_incomplete"
        possible_reuse = None
        match_strength = "incomplete_requires_review"

    return {
        "possible_reuse": possible_reuse,
        "status": status,
        "matched_reference": best["reference_path"].name if possible_reuse is True else None,
        "nearest_reference_candidate": best["reference_path"].name,
        "hash_distance": best["hash_distances"].get("phash"),
        "hash_distances": best["hash_distances"],
        "normalized_hash_distances": best["normalized_hash_distances"],
        "weighted_hash_distance": best["weighted_hash_distance"],
        "hash_similarity_score": best["hash_similarity_score"],
        "hash_match_votes": best["hash_match_votes"],
        "file_digest_match": best["file_digest_match"],
        "pixel_digest_match": best["pixel_digest_match"],
        "feature_good_matches": best["feature_good_matches"],
        "feature_inliers": best["feature_inliers"],
        "feature_inlier_ratio": best["feature_inlier_ratio"],
        "feature_status": best["feature_status"],
        # Kept for compatibility: this is deliberately null because the
        # thresholds have not been calibrated against labeled image pairs.
        "match_confidence": None,
        "match_strength": match_strength,
        "threshold": threshold,
        "reference_count": compared_count,
        "reference_collection_count": len(reference_files),
        "failed_references": failed_references,
        "method": "exact_sha256+canonical_pixels+phash+dhash+whash+average_hash+colorhash+sift",
        "calibration_status": "not_calibrated_without_labeled_pairs",
        "note": (
            "Heuristic comparison against the local reference collection only; "
            "match strength is not a calibrated probability or accuracy. A match does not establish "
            "the original event, date, location, or misleading intent."
        ),
    }
