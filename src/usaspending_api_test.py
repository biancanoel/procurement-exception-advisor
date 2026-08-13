"""Run a minimal contract-award search against the public USAspending API."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ENDPOINT = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
CONTRACT_AWARD_CODES = ["A", "B", "C", "D"]
FIELDS = [
    "Award ID",
    "Recipient Name",
    "Awarding Agency",
    "Award Amount",
    "Description",
    "Start Date",
    "End Date",
    "NAICS Code",
    "NAICS Description",
    "PSC Code",
    "PSC Description",
]


def build_request_body(keyword: str) -> dict[str, Any]:
    """Build the small, documented USAspending award-search request."""

    keyword = keyword.strip()
    if not keyword:
        raise ValueError("keyword must not be blank")

    return {
        "filters": {
            "award_type_codes": CONTRACT_AWARD_CODES,
            "keywords": [keyword],
        },
        "fields": FIELDS,
        "page": 1,
        "limit": 5,
        "sort": "Award Amount",
        "order": "desc",
    }


def search_contract_awards(keyword: str) -> tuple[int, dict[str, Any]]:
    """POST one contract-award search and return its status and JSON body."""

    request = Request(
        ENDPOINT,
        data=json.dumps(build_request_body(keyword)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            status = response.status
            raw_body = response.read()
    except HTTPError as error:
        print(f"HTTP status: {error.code}")
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"USAspending rejected the request: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"USAspending request failed: {error}") from error

    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("USAspending returned malformed JSON") from error

    if not isinstance(payload, dict):
        raise RuntimeError("USAspending returned an unexpected JSON response")
    return status, payload


def print_response(status: int, payload: dict[str, Any]) -> None:
    """Print a compact summary of the award-search response."""

    print(f"HTTP status: {status}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("USAspending response is missing a results list")

    print(f"Returned results: {len(results)}")
    pagination = payload.get("page_metadata", {})
    print(f"Pagination: {json.dumps(pagination, sort_keys=True)}")

    if not results:
        print("No contract awards matched the search.")
        return

    for rank, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            raise RuntimeError("USAspending returned an unexpected award result")

        amount = result.get("Award Amount")
        amount_text = f"${amount:,.2f}" if isinstance(amount, (int, float)) else "N/A"
        print(f"\n{rank}. Award ID: {result.get('Award ID') or 'N/A'}")
        print(f"   Recipient: {result.get('Recipient Name') or 'N/A'}")
        print(f"   Awarding agency: {result.get('Awarding Agency') or 'N/A'}")
        print(f"   Award amount: {amount_text}")
        print(f"   Description: {result.get('Description') or 'N/A'}")
        print(
            "   Dates: "
            f"{result.get('Start Date') or 'N/A'} to "
            f"{result.get('End Date') or 'N/A'}"
        )
        print(
            "   NAICS: "
            f"{result.get('NAICS Code') or 'N/A'} — "
            f"{result.get('NAICS Description') or 'N/A'}"
        )
        print(
            "   PSC: "
            f"{result.get('PSC Code') or 'N/A'} — "
            f"{result.get('PSC Description') or 'N/A'}"
        )


def main() -> int:
    """Search USAspending for up to five contract awards."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "keyword",
        nargs="?",
        default="emergency",
        help="Award keyword to search (default: emergency)",
    )
    args = parser.parse_args()

    try:
        status, payload = search_contract_awards(args.keyword)
        print_response(status, payload)
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
