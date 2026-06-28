import asyncio
import logging
import os

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Subreddits most likely to surface B2B buying signals
_SIGNAL_SUBREDDITS = [
    "entrepreneur", "startups", "SaaS", "smallbusiness",
    "hiring", "forhire", "venturecapital", "technology",
]


def _get_reddit():
    """Lazy PRAW Reddit client — reads credentials from environment."""
    import praw
    return praw.Reddit(
        client_id    =os.getenv("REDDIT_CLIENT_ID", ""),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
        user_agent   ="lead-gen-agent/1.0",
    )


def _post_to_signal(post, company_name: str, source_sub: str) -> dict | None:
    """Maps a PRAW Submission to a SignalData-shaped dict. Returns None if too short."""
    text = f"{post.title} {post.selftext or ''}"
    if len(text.strip()) < 30:
        return None
    # Rough strength heuristic: upvote ratio + comment count
    strength = "high" if (post.score > 100 or post.num_comments > 20) else "medium" if post.score > 20 else "low"
    return {
        "company" : company_name,
        "signal"  : post.title[:200],
        "source"  : f"reddit/r/{source_sub}",
        "strength": strength,
        "url"     : f"https://www.reddit.com{post.permalink}",
    }


@tool
async def search_reddit_signals(
    company_name: str,
    keywords: list[str],
    max_results: int = 10,
) -> list[dict]:
    """
    Searches Reddit for buying signals related to a company or topic keywords.
    Checks curated B2B subreddits for posts mentioning the company or keywords.
    Returns SignalData-shaped dicts with source='reddit/r/<subreddit>', strength (low/medium/high).
    Returns [] if PRAW credentials are missing or search fails.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    if not client_id:
        logger.warning("REDDIT_CLIENT_ID not set — Reddit signal search skipped.")
        return []

    query = f"{company_name} {' '.join(keywords)}"
    signals: list[dict] = []

    def _search_sync() -> list[dict]:
        reddit = _get_reddit()
        results = []
        try:
            for sub_name in _SIGNAL_SUBREDDITS:
                subreddit = reddit.subreddit(sub_name)
                for post in subreddit.search(query, limit=3, sort="relevance"):
                    signal = _post_to_signal(post, company_name, sub_name)
                    if signal:
                        results.append(signal)
                    if len(results) >= max_results:
                        return results
        except Exception as e:
            logger.warning("Reddit search failed: %s", e)
        return results

    try:
        signals = await asyncio.to_thread(_search_sync)
    except Exception as e:
        logger.warning("Reddit thread failed: %s", e)

    logger.info("reddit_tool: found %d signal(s) for '%s'.", len(signals), company_name)
    return signals[:max_results]
