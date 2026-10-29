"""Build search drafts from source claims. Does not search or verify claims."""

from copy import deepcopy
import re


class QueryGenerator:
    # Only articles are removed in the alternate keyword query. Retain
    # negation, reporting words, quantities, dates, and qualifications.
    ENGLISH_ARTICLES = {"a", "an", "the"}

    def generate(self, claim, language=None):
        text = re.sub(r"\s+", " ", claim.get("text", "")).strip()
        language = language or claim.get("entities", {}).get("language")
        if language is None and text:
            language = "si" if re.search(r"[\u0D80-\u0DFF]", text) else "en"

        screening = claim.get("review_screening_status", "not_performed")
        # Flags must never be silently treated as approval for searching.
        if not text:
            status = "empty_claim"
        elif claim.get("needs_review"):
            status = "draft_needs_review"
        elif screening != "heuristic_screened":
            status = "draft_unscreened"
        else:
            status = "draft_screened"

        queries = []
        if text:
            queries.append({"kind": "source_text", "query": text})
            # Keep full Sinhala text. A word-deletion heuristic can break
            # Sinhala relationships and does not establish key concepts.
            if language == "en":
                tokens = text.split()
                keywords = " ".join(
                    word for word in tokens
                    if word.strip(".,;:!?\"“”()[]").casefold() not in self.ENGLISH_ARTICLES
                )
                if keywords and keywords != text:
                    queries.append({"kind": "article_reduced", "query": keywords})

        return {
            "source_claim_id": claim.get("claim_id"),
            "source_text": claim.get("text", ""),
            "language": language,
            "status": status,
            "review_screening_status": screening,
            "needs_review": claim.get("needs_review"),
            "review_reasons": deepcopy(claim.get("review_reasons", [])),
            "queries": queries,
        }

    def enrich_claims(self, claims, language=None):
        """Add drafts without changing source claims or their metadata."""
        enriched = deepcopy(claims)
        for claim in enriched:
            claim["search_queries"] = self.generate(claim, language)
        return enriched
