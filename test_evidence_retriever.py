from app.services.evidence_retriever import (
    EvidenceRetriever
)


test_query_record = {
    "source_claim_id": 1,

    "source_text":
        "The Sri Lankan government "
        "announced a change to VAT.",

    "language": "en",

    "queries": [
        {
            "kind": "source_text",
            "query":
                "Sri Lanka government VAT"
        },
        {
            "kind": "article_reduced",
            "query":
                "Sri Lanka VAT government"
        }
    ]
}


print(
    "\n================================="
)

print(
    "EVIDENCE RETRIEVER TEST"
)

print(
    "=================================\n"
)


retriever = EvidenceRetriever()


result = retriever.retrieve_for_query_record(
    test_query_record,
    results_per_query=5
)


print(
    "\nRetrieval Status:",
    result["retrieval_status"]
)

print(
    "Evidence Found:",
    result["evidence_count"]
)


for evidence in result["evidence"]:

    print(
        "\n---------------------------------"
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