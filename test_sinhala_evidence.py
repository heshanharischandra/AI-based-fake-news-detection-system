import json
from pathlib import Path

from app.services.evidence_retriever import EvidenceRetriever


RESULT_PATH = Path("result.json")


print("\n========================================")
print("SINHALA EVIDENCE RETRIEVAL TEST")
print("========================================\n")


if not RESULT_PATH.exists():
    raise FileNotFoundError(
        f"Could not find: {RESULT_PATH}"
    )


with open(
    RESULT_PATH,
    "r",
    encoding="utf-8"
) as file:

    pipeline_result = json.load(file)


search_queries = pipeline_result.get(
    "search_queries"
)


if not search_queries:
    raise ValueError(
        "No search_queries found in result.json."
    )


# Your JSON may contain either:
# one query record as a dictionary
# or several query records as a list.

if isinstance(search_queries, dict):
    query_records = [
        search_queries
    ]

elif isinstance(search_queries, list):
    query_records = search_queries

else:
    raise ValueError(
        "search_queries has an unexpected format."
    )


retriever = EvidenceRetriever()


for query_record in query_records:

    language = query_record.get(
        "language"
    )

    claim_id = query_record.get(
        "source_claim_id"
    )

    claim_text = query_record.get(
        "source_text",
        ""
    )


    print("\n----------------------------------------")
    print("CLAIM INFORMATION")
    print("----------------------------------------")

    print(
        "Claim ID:",
        claim_id
    )

    print(
        "Language:",
        language
    )

    print(
        "Claim:",
        claim_text
    )


    if language != "si":

        print(
            "\nWARNING:"
            " This query record is not marked as Sinhala."
        )


    print(
        "\nStarting evidence retrieval..."
    )


    result = retriever.retrieve_for_query_record(
        query_record,
        results_per_query=5
    )


    print(
        "\n========================================"
    )

    print(
        "RETRIEVAL RESULT"
    )

    print(
        "========================================"
    )


    print(
        "Retrieval Status:",
        result["retrieval_status"]
    )

    print(
        "Evidence Found:",
        result["evidence_count"]
    )

    print(
        "Trusted Evidence:",
        result.get(
            "trusted_evidence_count",
            0
        )
    )

    print(
        "Needs Review:",
        result.get(
            "review_required_count",
            0
        )
    )


    for evidence in result["evidence"]:

        print(
            "\n----------------------------------------"
        )

        print(
            "Evidence ID:",
            evidence["evidence_id"]
        )

        print(
            "Source:",
            evidence["source"]
        )

        print(
            "Trusted:",
            evidence["trusted_source"]
        )

        print(
            "Status:",
            evidence["evidence_status"]
        )

        print(
            "Title:",
            evidence["title"]
        )

        print(
            "Date:",
            evidence["published_at"]
        )

        print(
            "URL:",
            evidence["url"]
        )

        print(
            "Description:",
            evidence["description"]
        )