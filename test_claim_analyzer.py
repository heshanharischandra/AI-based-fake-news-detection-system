from app.services.claim_analyzer import ClaimAnalyzer


text = """
2026 ඔක්තෝබර් 10 දින කොළඹදී සිද්ධිය සිදු විය.
පොලිසිය සැකකරුවෙකු අත්අඩංගුවට ගෙන ඇත.
පුද්ගලයන් තිදෙනෙකු රෝහල්ගත කර ඇත.
"""

analyzer = ClaimAnalyzer()

result = analyzer.analyze(
    transcript=text,
    language="si"
)

print(
    "\nCheck-worthy claims found:",
    result["claim_count"]
)

for claim in result["claims"]:

    print(
        f"\nClaim {claim['claim_id']}:"
    )

    print(claim["text"])

    print(
        "Score:",
        claim["check_worthy_score"]
    )

    print(
        "Reasons:",
        claim["reasons"]
    )