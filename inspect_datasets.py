from pathlib import Path
import pandas as pd


print("\n====================================")
print("DATASET INSPECTION")
print("====================================")


# ==================================================
# 1. SINHALA MISINFORMATION DATASET
# ==================================================

print("\n====================================")
print("1. SINHALA MISINFORMATION DATASET")
print("====================================")

sinhala_path = Path(
    "data/datasets/SinhalaMisinformation/Corpus.csv"
)

if sinhala_path.exists():

    sinhala = pd.read_csv(
        sinhala_path,
        encoding="utf-8",
        encoding_errors="replace"
    )

    print("Status: FOUND")
    print("Rows:", len(sinhala))
    print("Columns:", sinhala.columns.tolist())

    print("\nSample:")
    print(sinhala.head())

else:

    print("Status: NOT FOUND")
    print("Path:", sinhala_path)


# ==================================================
# 2. ISOT TRUE NEWS
# ==================================================

print("\n====================================")
print("2. ISOT ENGLISH TRUE NEWS")
print("====================================")

true_path = Path(
    "data/datasets/ISOT_English/True.csv"
)

if true_path.exists():

    true_news = pd.read_csv(
        true_path,
        encoding_errors="replace"
    )

    print("Status: FOUND")
    print("Rows:", len(true_news))
    print("Columns:", true_news.columns.tolist())

    print("\nSample:")
    print(true_news.head())

else:

    print("Status: NOT FOUND")


# ==================================================
# 3. ISOT FAKE NEWS
# ==================================================

print("\n====================================")
print("3. ISOT ENGLISH FAKE NEWS")
print("====================================")

fake_path = Path(
    "data/datasets/ISOT_English/Fake.csv"
)

if fake_path.exists():

    fake_news = pd.read_csv(
        fake_path,
        encoding_errors="replace"
    )

    print("Status: FOUND")
    print("Rows:", len(fake_news))
    print("Columns:", fake_news.columns.tolist())

    print("\nSample:")
    print(fake_news.head())

else:

    print("Status: NOT FOUND")


# ==================================================
# 4. VICLAIM
# ==================================================

print("\n====================================")
print("4. VICLAIM VIDEO CLAIM DATASET")
print("====================================")

viclaim_path = Path(
    "data/datasets/ViClaim/data/viclaim.csv"
)

if viclaim_path.exists():

    viclaim = pd.read_csv(
        viclaim_path,
        encoding="utf-8",
        encoding_errors="replace"
    )

    print("Status: FOUND")
    print("Rows:", len(viclaim))
    print("Columns:", viclaim.columns.tolist())

    print("\nSample:")
    print(viclaim.head())

else:

    print("Status: NOT FOUND")


print("\n====================================")
print("DATASET INSPECTION COMPLETED")
print("====================================")