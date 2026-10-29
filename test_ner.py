from app.services.ner_extractor import NERExtractor


# =========================================================
# LOAD NER MODEL
# =========================================================

ner = NERExtractor()


# =========================================================
# ENGLISH NER TEST
# =========================================================

print("\n==============================")
print("ENGLISH NER TEST")
print("==============================")

english_text = (
    "Major General Suresh Saleh was admitted "
    "to Colombo National Hospital in 2019."
)

english_result = ner.extract(
    english_text,
    language="en"
)

print("\nInput:")
print(english_text)

print("\nResult:")
print(english_result)


# =========================================================
# SINHALA NER TEST
# =========================================================

print("\n==============================")
print("SINHALA NER TEST")
print("==============================")

sinhala_text = (
    "2026 ඔක්තෝබර් 10 දින කොළඹදී සිද්ධිය සිදු විය. "
    "අද පොලිසිය සැකකරුවෙකු අත්අඩංගුවට ගෙන ඇත. "
    "සෙනසුරාදා කොළඹදී උත්සවයක් පැවැත්විණි. "
    "මේ වසරේ ශ්‍රී ලංකාවේ සිදුවීම් කිහිපයක් වාර්තා වී ඇත."
)

sinhala_result = ner.extract(
    sinhala_text,
    language="si"
)

print("\nInput:")
print(sinhala_text)

print("\nResult:")
print(sinhala_result)


# =========================================================
# DISPLAY SINHALA RESULT CLEARLY
# =========================================================

print("\n==============================")
print("SINHALA EXTRACTED ENTITIES")
print("==============================")

print(
    "Persons:",
    sinhala_result["persons"]
)

print(
    "Organizations:",
    sinhala_result["organizations"]
)

print(
    "Locations:",
    sinhala_result["locations"]
)

print(
    "Dates / Time Expressions:",
    sinhala_result["dates"]
)

print(
    "Numbers:",
    sinhala_result["numbers"]
)

print(
    "Miscellaneous:",
    sinhala_result["miscellaneous"]
)