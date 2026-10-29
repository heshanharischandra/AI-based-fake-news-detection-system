import os
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


load_dotenv()


class EvidenceRetriever:

    NEWSAPI_URL = "https://newsapi.org/v2/everything"

    TRUSTED_DOMAINS = [
        "reuters.com",
        "bbc.com",
        "bbc.co.uk",
        "adaderana.lk",
        "hirunews.lk"
    ]

    def __init__(self):
        self.api_key = os.getenv("NEWSAPI_KEY")

        if not self.api_key:
            raise ValueError(
                "NEWSAPI_KEY was not found. "
                "Add it to your .env file."
            )

    # --------------------------------------------------
    # DOMAIN HELPERS
    # --------------------------------------------------

    def _get_domain(self, url):

        if not url:
            return ""

        try:
            domain = urlparse(url).netloc.lower()

            if domain.startswith("www."):
                domain = domain[4:]

            return domain

        except Exception:
            return ""

    def _is_trusted_domain(self, url):

        domain = self._get_domain(url)

        for trusted_domain in self.TRUSTED_DOMAINS:

            if (
                domain == trusted_domain
                or domain.endswith(
                    "." + trusted_domain
                )
            ):
                return True

        return False

    # --------------------------------------------------
    # NEWSAPI REQUEST
    # --------------------------------------------------

    def _request_newsapi(
        self,
        query,
        page_size=5,
        trusted_only=False
    ):

        if not query:
            return []

        params = {
            "q": query[:500],
            "sortBy": "relevancy",
            "pageSize": page_size
        }

        # Apply domain restriction only when requested
        if trusted_only:

            params["domains"] = ",".join(
                self.TRUSTED_DOMAINS
            )

        headers = {
            "X-Api-Key": self.api_key
        }

        try:

            response = requests.get(
                self.NEWSAPI_URL,
                params=params,
                headers=headers,
                timeout=20
            )

            response.raise_for_status()

            data = response.json()

        except requests.RequestException as error:

            print(
                "Evidence retrieval request failed:",
                error
            )

            return []

        # Debug information
        print(
            "NewsAPI status:",
            data.get("status")
        )

        print(
            "NewsAPI total results:",
            data.get(
                "totalResults",
                0
            )
        )

        if data.get("status") != "ok":

            print(
                "NewsAPI returned an error:",
                data.get("message")
            )

            return []

        return data.get(
            "articles",
            []
        )

    # --------------------------------------------------
    # FORMAT ARTICLE
    # --------------------------------------------------

    def _format_article(
        self,
        article,
        retrieval_method
    ):

        url = article.get("url")

        if not url:
            return None

        source_data = (
            article.get("source")
            or {}
        )

        trusted = self._is_trusted_domain(
            url
        )

        if trusted:
            evidence_status = (
                "trusted_candidate"
            )
        else:
            evidence_status = (
                "candidate_requires_review"
            )

        return {
            "title": (
                article.get("title")
                or ""
            ),

            "source": (
                source_data.get("name")
                or self._get_domain(url)
            ),

            "domain": self._get_domain(
                url
            ),

            "url": url,

            "published_at": (
                article.get(
                    "publishedAt"
                )
            ),

            "description": (
                article.get(
                    "description"
                )
                or ""
            ),

            "content": (
                article.get(
                    "content"
                )
                or ""
            ),

            "retrieval_method":
                retrieval_method,

            "trusted_source":
                trusted,

            "evidence_status":
                evidence_status
        }

    # --------------------------------------------------
    # SEARCH NEWSAPI
    # --------------------------------------------------

    def search_newsapi(
        self,
        query,
        page_size=5
    ):

        if not query:
            return []

        print(
            "\n[1] Searching trusted domains..."
        )

        trusted_articles = (
            self._request_newsapi(
                query=query,
                page_size=page_size,
                trusted_only=True
            )
        )

        results = []

        seen_urls = set()

        # --------------------------------------------------
        # FIRST: trusted-domain results
        # --------------------------------------------------

        for article in trusted_articles:

            formatted = self._format_article(
                article,
                retrieval_method=(
                    "newsapi_trusted_domains"
                )
            )

            if not formatted:
                continue

            url = formatted["url"]

            if url in seen_urls:
                continue

            seen_urls.add(url)

            results.append(
                formatted
            )

        if results:

            print(
                "Trusted evidence candidates found:",
                len(results)
            )

            return results

        # --------------------------------------------------
        # FALLBACK: broad NewsAPI search
        # --------------------------------------------------

        print(
            "\nNo trusted-domain results found."
        )

        print(
            "[2] Trying broader NewsAPI search..."
        )

        broad_articles = (
            self._request_newsapi(
                query=query,
                page_size=max(
                    page_size * 2,
                    10
                ),
                trusted_only=False
            )
        )

        for article in broad_articles:

            formatted = self._format_article(
                article,
                retrieval_method=(
                    "newsapi_broad_fallback"
                )
            )

            if not formatted:
                continue

            url = formatted["url"]

            if url in seen_urls:
                continue

            seen_urls.add(url)

            results.append(
                formatted
            )

            # Do not return too many
            if len(results) >= page_size:
                break

        print(
            "Broad evidence candidates found:",
            len(results)
        )

        return results

    # --------------------------------------------------
    # RETRIEVE FOR QUERY-GENERATOR OUTPUT
    # --------------------------------------------------

    def retrieve_for_query_record(
        self,
        query_record,
        results_per_query=5
    ):

        claim_id = query_record.get(
            "source_claim_id"
        )

        claim_text = query_record.get(
            "source_text",
            ""
        )

        language = query_record.get(
            "language"
        )

        query_candidates = (
            query_record.get("queries")
            or []
        )

        all_evidence = []

        seen_urls = set()

        attempted_queries = []

        # --------------------------------------------------
        # SEARCH EACH GENERATED QUERY
        # --------------------------------------------------

        for query_item in query_candidates:

            query_text = (
                query_item.get("query")
                or ""
            ).strip()

            query_kind = (
                query_item.get("kind")
                or "unknown"
            )

            if not query_text:
                continue

            attempted_queries.append({
                "kind": query_kind,
                "query": query_text
            })

            print(
                "\n================================="
            )

            print(
                "Searching evidence"
            )

            print(
                "================================="
            )

            print(
                "Query type:",
                query_kind
            )

            print(
                "Query:",
                query_text
            )

            search_results = (
                self.search_newsapi(
                    query=query_text,
                    page_size=(
                        results_per_query
                    )
                )
            )

            # --------------------------------------------------
            # MERGE RESULTS
            # --------------------------------------------------

            for article in search_results:

                url = article.get(
                    "url"
                )

                if not url:
                    continue

                if url in seen_urls:
                    continue

                seen_urls.add(
                    url
                )

                article[
                    "query_kind"
                ] = query_kind

                article[
                    "search_query"
                ] = query_text

                all_evidence.append(
                    article
                )

        # --------------------------------------------------
        # ADD EVIDENCE IDS
        # --------------------------------------------------

        for index, evidence in enumerate(
            all_evidence,
            start=1
        ):

            evidence[
                "evidence_id"
            ] = f"E{index:03d}"

        # --------------------------------------------------
        # COUNT TRUSTED / UNTRUSTED
        # --------------------------------------------------

        trusted_count = sum(
            1
            for item in all_evidence
            if item.get(
                "trusted_source"
            )
        )

        review_count = (
            len(all_evidence)
            - trusted_count
        )

        # --------------------------------------------------
        # RETRIEVAL STATUS
        # --------------------------------------------------

        if trusted_count > 0:

            retrieval_status = (
                "trusted_candidate_evidence_found"
            )

        elif all_evidence:

            retrieval_status = (
                "candidate_evidence_requires_review"
            )

        else:

            retrieval_status = (
                "no_candidate_evidence"
            )

        # --------------------------------------------------
        # RETURN
        # --------------------------------------------------

        return {
            "source_claim_id":
                claim_id,

            "source_text":
                claim_text,

            "language":
                language,

            "retrieval_status":
                retrieval_status,

            "attempted_queries":
                attempted_queries,

            "evidence_count":
                len(all_evidence),

            "trusted_evidence_count":
                trusted_count,

            "review_required_count":
                review_count,

            "evidence":
                all_evidence
        }