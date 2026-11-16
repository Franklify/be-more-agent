import re
import sys

if sys.version_info >= (3, 10):
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException
else:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import DuckDuckGoSearchException as DDGSException


def search_web(query, news=False):
    if not isinstance(query, str) or not query.strip():
        return "INVALID_ACTION"

    query = query.strip()
    methods = ("news", "text") if news else ("text",)
    failed = False
    with DDGS(timeout=5) as client:
        for method in methods:
            try:
                results = list(getattr(client, method)(query, region="us-en", max_results=3))
            except DDGSException as exc:
                if str(exc).strip().lower() == "no results found.":
                    results = []
                else:
                    print(f"Search {method} error: {exc}", flush=True)
                    failed = True
                    continue
            if results:
                lines = [f"SEARCH RESULTS for '{query}':"]
                for result in results:
                    title = " ".join(str(result.get("title") or "Untitled").split())
                    body = " ".join(str(result.get("body") or result.get("snippet") or "").split())[:400]
                    url = " ".join(str(result.get("href") or result.get("url") or "").split())
                    date = " ".join(str(result.get("date") or "").split())
                    lines.append(f"Title: {title}\nDate: {date}\nURL: {url}\nSnippet: {body}")
                return "\n\n".join(lines)
    return "SEARCH_ERROR" if failed else "SEARCH_EMPTY"


def is_news_query(text):
    return bool(re.search(r"\b(news|headlines|current events)\b", text, re.IGNORECASE))
