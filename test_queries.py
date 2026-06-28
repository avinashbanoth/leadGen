# -*- coding: utf-8 -*-
"""
Manual end-to-end query test -- runs 10 diverse queries against the FastAPI server
and prints structured results for evaluation.

Run the server first:  uvicorn api.main:app --port 8000
Then: python test_queries.py
"""

import asyncio
import time
import httpx

API_URL = "http://localhost:8000/chat"

QUERIES = [
    # 1. Clean specific lead-gen query
    "Find CTOs at Series B SaaS companies in the US that use React",
    # 2. UK + specific role
    "I need VP of Engineering contacts at UK fintech startups with 50-200 employees",
    # 3. Signal-driven query
    "Find founders of AI startups that recently raised funding and are hiring ML engineers",
    # 4. Vague -- should trigger clarification
    "find me some leads",
    # 5. Non-lead-gen -- should be rejected
    "What is the weather in London today?",
    # 6. Non-lead-gen disguised -- should be rejected
    "Write me a poem about sales",
    # 7. Specific industry + role + geography
    "CFOs at healthcare software companies in Germany with revenue over 10M",
    # 8. Tech stack filter
    "Find heads of DevOps at companies using Kubernetes and AWS in Singapore",
    # 9. Signal-heavy query
    "Founders at cybersecurity startups that posted on HN or raised Series A in 2024",
    # 10. Role expansion test
    "HR heads at mid-size manufacturing companies in India",
]


async def run_query(client: httpx.AsyncClient, idx: int, query: str) -> dict:
    start = time.time()
    try:
        response = await client.post(
            API_URL,
            json={"query": query},
            timeout=120.0,
        )
        elapsed = round(time.time() - start, 1)
        if response.status_code == 200:
            data = response.json()
            return {
                "idx"    : idx,
                "query"  : query,
                "status" : data.get("status"),
                "message": data.get("message", "")[:300],
                "stats"  : data.get("stats", {}),
                "errors" : data.get("errors", [])[:3],
                "elapsed": elapsed,
                "http"   : 200,
            }
        else:
            return {
                "idx"    : idx,
                "query"  : query,
                "http"   : response.status_code,
                "elapsed": elapsed,
                "error"  : response.text[:200],
            }
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        return {"idx": idx, "query": query, "http": 0, "elapsed": elapsed, "error": str(e)}


async def main():
    SEP = "=" * 70
    DIV = "-" * 70
    print(SEP)
    print("LEAD GEN AGENT -- 10-QUERY EVALUATION")
    print(SEP)

    async with httpx.AsyncClient() as client:
        try:
            health = await client.get("http://localhost:8000/health", timeout=5)
            print(f"Server: {'UP' if health.status_code == 200 else 'DOWN'}\n")
        except Exception:
            print("Server is NOT running. Start it with: python -m uvicorn api.main:app --port 8000")
            return

        for i, query in enumerate(QUERIES, 1):
            print(f"\n{DIV}")
            print(f"Query {i:02d}: {query}")
            print(DIV)

            result = await run_query(client, i, query)

            print(f"Status  : {result.get('status', result.get('error', 'N/A'))}")
            print(f"HTTP    : {result.get('http')}  |  Time: {result.get('elapsed')}s")

            stats = result.get("stats", {})
            if stats:
                print(
                    f"Contacts: {stats.get('total', 0)} total -- "
                    f"{stats.get('verified', 0)} verified / "
                    f"{stats.get('partial', 0)} partial / "
                    f"{stats.get('not_found', 0)} not_found"
                )

            errors = result.get("errors", [])
            if errors:
                print(f"Errors  : {len(errors)} logged")
                for e in errors:
                    print(f"  * {str(e)[:120]}")

            msg = result.get("message", "")
            if msg:
                print(f"Response: {msg[:250]}")

    print(f"\n{SEP}")
    print("EVALUATION COMPLETE")
    print(SEP)


if __name__ == "__main__":
    asyncio.run(main())
