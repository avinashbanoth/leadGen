import asyncio
import json
import logging
import os
import re

from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import GraphState
from tools.searxng_tool import searxng_search
from tools.crawl4ai_tool import scrape_company_website
from tools.overpass_tool import search_overpass_businesses

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Two Groq LLMs:
#   heavy (70b) — name extraction (needs higher accuracy)
#   light (8b)  — industry inference (high-volume, simple classification)
# ---------------------------------------------------------------------------

_llm_heavy = None
_llm_light = None


def _get_heavy_llm():
    global _llm_heavy
    if _llm_heavy is None:
        from langchain_groq import ChatGroq
        _llm_heavy = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    return _llm_heavy


def _get_llm():
    global _llm_light
    if _llm_light is None:
        from langchain_groq import ChatGroq
        _llm_light = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    return _llm_light


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_EXTRACT_NAMES_PROMPT = """You are a B2B researcher. The text below was scraped from a webpage.
Your job is to extract SPECIFIC company brand names from it.

Rules:
- Return ONLY proper company brand names (e.g. "Acme Corp", "Vertex Systems", "NovaTech", "BluePeak")
- Each name must be a specific company, NOT a generic category word
- Do NOT include generic words like: "Technology", "Solutions", "Startup", "Company", "Software"
  (these are category words, not company names)
- Each name should be at least 2 words long OR a unique brand word (not a plain dictionary word)
- Return 5–10 names maximum
- Return ONLY a JSON array of strings. No explanation.

Good output: ["Acme Corp", "Vertex Systems", "NovaTech", "BluePeak", "Meridian Software"]
Bad output: ["Technology", "Solutions", "Startup", "Company", "Software"]

Text:
{text}"""

_INDUSTRY_PROMPT = """You are a strict B2B analyst. Evaluate this web page and return JSON.

STEP 1 — Is this a real company website?
A real company website promotes its OWN products or services to customers.
IMMEDIATELY set matches=false for ANY of these:
- Wikipedia, dictionary, encyclopedia pages
- News articles or press releases
- Tutorial or educational pages ("What is X?", "Understanding X")
- Government authority pages (unless selling commercial products)
- Startup/company directories, aggregator sites (Crunchbase, Tracxn, etc.)
- List articles ("Top 10...", "Best X companies...")
- Generic resource or blog pages (/blog/, /resources/, /guides/, /news/)
- Freelancer or job marketplaces

STEP 2 — Does this company match the target criteria?
Target criteria: {criteria}

Evaluate STRICTLY against the criteria above — the criteria defines the industry and location.
Do NOT apply any fixed rules about which industries "count" — use only the criteria provided.
Any legitimate business that matches the stated industry and location qualifies.

If the criteria specifies a LOCATION:
- The company must be BASED IN or primarily serving that location
- A company merely having an office or satellite team in that location does NOT qualify
- If you cannot determine the company's primary location from the page content, set matches=false

If the criteria specifies an INDUSTRY:
- The company must operate in that sector as its primary business
- A generic "software development" or "IT services" company does NOT match a specific sector
  unless the page clearly shows domain-specific products (e.g. a healthcare-specific SaaS)

Website content (first 2000 chars):
{raw_text}

Return ONLY this JSON (no explanation):
{{"industry": "1-3 word industry label", "matches": true/false, "reason": "one sentence"}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Generic words that make useless company names when extracted from a domain
_GENERIC_DOMAIN_WORDS = {
    "fintech", "saas", "software", "tech", "technology", "digital", "cloud",
    "solutions", "services", "company", "enterprise", "business", "startup",
    "news", "blog", "media", "info", "portal", "online", "web", "app",
    "product", "industry", "industrial", "market", "global", "group",
}

# Subdomain prefixes that indicate non-company pages
_SUBDOMAIN_PREFIXES = {
    "assets", "api", "cdn", "static", "img", "images", "files", "media",
    "docs", "help", "support", "status", "blog", "shop", "store",
    "mail", "ftp", "dev", "staging", "test", "sandbox", "demo", "app",
    "portal", "partnerfinder", "partners", "community", "forum", "careers",
    "jobs", "login", "auth", "account", "secure", "pay", "checkout",
    "m", "mobile", "amp", "wap", "www2", "go", "news", "resource", "resources",
}

# URL path first-segments that indicate a list/article page
_ARTICLE_PATH_PREFIXES = {
    "articles", "article", "blog", "blogs", "news", "insights",
    "learn", "learning", "guides", "guide", "resources", "resource",
    "lists", "list", "rankings", "ranking", "top", "best", "versus",
    "reviews", "review", "compare", "comparison", "directory",
    "startup", "startups", "companies", "markets",
    "categories", "category", "sector", "sectors",
}

# Domains to skip outright
_SKIP_DOMAINS = {
    "linkedin.com", "reddit.com", "quora.com", "medium.com", "substack.com",
    "youtube.com", "twitter.com", "facebook.com", "instagram.com",
    "wikipedia.org", "dictionary.com", "merriam-webster.com", "investopedia.com",
    "cgaa.org", "britannica.com", "wikihow.com", "wikidata.org",
    "shopify.com", "amazon.com", "flipkart.com", "ebay.com",
    "timesofindia.com", "economictimes.com", "livemint.com", "businessinsider.com",
    "techcrunch.com", "venturebeat.com", "forbes.com", "businesswire.com",
    "prnewswire.com", "globenewswire.com", "inc42.com", "yourstory.com",
    "moneycontrol.com", "mint.com", "thehindu.com", "ndtv.com",
    "crunchbase.com", "startupindia.gov.in", "tracxn.com", "angellist.com",
    "clutch.co", "g2.com", "capterra.com", "glassdoor.com", "indeed.com",
    "fintech.com", "thebalance.com", "nerdwallet.com", "bankrate.com",
    "tripadvisor.com", "tripadvisor.in", "makemytrip.com", "booking.com",
    "airbnb.com", "expedia.com", "hotels.com", "agoda.com",
    "naukri.com", "shine.com", "monster.com", "foundit.in",
    "builtin.com", "builtinnyc.com", "builtinboston.com", "builtinbengaluru.in",
    "marketwatch.com", "bloomberg.com", "reuters.com", "wsj.com",
    "ft.com", "barrons.com", "yahoo.com",
    "beststartup.in", "beststartup.us", "18startup.com", "buzz4ai.com",
    "startupranking.com", "startupbonsai.com", "f6s.com",
    "ynos.in", "tofler.in", "zaubacorp.com",
    "tradebrains.in", "entrackr.com", "afaqs.com",
    "worldbank.org",
}

# Domains to skip in Phase 1 (content is never a company-name list)
_PHASE1_SKIP_DOMAINS = {
    "github.com", "gitlab.com", "stackoverflow.com", "stackexchange.com",
    "zhihu.com", "weibo.com", "baidu.com", "csdn.net",
    "youtube.com", "reddit.com", "twitter.com", "facebook.com",
    "instagram.com", "tiktok.com", "snapchat.com", "pinterest.com",
    "linkedin.com", "quora.com", "medium.com", "substack.com",
    "wikipedia.org", "britannica.com", "wikihow.com", "wikidata.org",
    "glassdoor.com", "indeed.com", "naukri.com", "monster.com",
    "bloomberg.com", "reuters.com", "wsj.com", "ft.com",
    "investopedia.com", "thebalance.com", "nerdwallet.com",
}


def _root_domain(domain: str) -> str:
    """Returns the root name from a domain (strips www. and TLD), used for deduplication."""
    d = domain[4:] if domain.startswith("www.") else domain
    parts = d.split(".")
    return parts[-2] if len(parts) >= 2 else d


def _clean_company_name(domain: str, fallback_title: str) -> str:
    """Derives a clean company name from the domain."""
    bare = domain.replace("www.", "").split(".")[0].lower()
    if len(bare) >= 2 and bare not in _GENERIC_DOMAIN_WORDS:
        return bare.upper()
    cleaned = re.split(r'[|\-–—:]', fallback_title)[0].strip()
    return cleaned or fallback_title


def _build_criteria_str(company_filters: dict) -> str:
    parts = []
    if company_filters.get("industry"):
        parts.append(f"Industry: {company_filters['industry']}")
    if company_filters.get("keywords"):
        parts.append(f"Keywords: {', '.join(company_filters['keywords'])}")
    if company_filters.get("revenue_min"):
        parts.append(f"Minimum revenue: {company_filters['revenue_min']}")
    if company_filters.get("location"):
        parts.append(f"Location: {company_filters['location']}")
    if company_filters.get("company_size"):
        parts.append(f"Company size: {company_filters['company_size']}")
    if company_filters.get("tech_stack"):
        parts.append(f"Tech stack: {', '.join(company_filters['tech_stack'])}")
    return "\n".join(parts) or "No specific filters — find relevant companies."


def _extract_location_terms(criteria_str: str) -> list[str]:
    """Pulls location variants for soft presence checks (e.g. Bangalore → ['bangalore','bengaluru','india'])."""
    terms: list[str] = []
    for line in criteria_str.splitlines():
        if line.lower().startswith("location:"):
            loc = line.split(":", 1)[1].strip().lower()
            if loc:
                terms.append(loc)
                if loc in ("bangalore", "bengaluru"):
                    terms += ["bangalore", "bengaluru"]
                elif loc in ("mumbai", "bombay"):
                    terms += ["mumbai", "bombay"]
                elif loc == "delhi":
                    terms += ["delhi", "new delhi", "ncr"]
                if any(x in loc for x in ("india", "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "pune", "chennai")):
                    terms.append("india")
    return list(set(terms))


def _is_company_url(url: str, seen_roots: set) -> bool:
    """
    Returns True if the URL is worth scraping as a potential company homepage.
    Applies domain, subdomain, and path-pattern blocklists.
    """
    if not url.startswith("http"):
        return False
    domain = url.split("/")[2]
    root = _root_domain(domain)

    if root in seen_roots:
        return False
    if root.lower() in _GENERIC_DOMAIN_WORDS:
        return False
    if any(skip in domain for skip in _SKIP_DOMAINS):
        return False

    # Deep subdomain check (e.g. assets.new.siemens.com)
    bare = domain[4:] if domain.startswith("www.") else domain
    parts = bare.split(".")
    if len(parts) > 3:
        return False
    if len(parts) == 3 and parts[0] in _SUBDOMAIN_PREFIXES:
        return False

    # URL path checks
    url_path = url.split("/", 3)[3] if url.count("/") >= 3 else ""
    first_seg = url_path.split("/")[0].lower().split("?")[0] if url_path else ""

    if first_seg in _ARTICLE_PATH_PREFIXES:
        return False
    # Numeric-prefixed slugs like "101-bangalore-fintech-companies"
    if first_seg and first_seg[0].isdigit():
        return False
    # Slugs containing "-companies-" or "-startups-"
    if "-companies-" in url_path.lower() or "companies-in-" in url_path.lower():
        return False
    if "-startups-" in url_path.lower() or "startups-in-" in url_path.lower():
        return False
    # 3+ deep paths are resource/blog pages
    depth = len([p for p in url_path.split("/") if p])
    if depth >= 3:
        return False

    return True


async def _infer_industry(raw_text: str, criteria_str: str) -> tuple[str, bool]:
    """Uses 8b LLM to infer industry and check if the company matches criteria."""
    prompt = _INDUSTRY_PROMPT.format(criteria=criteria_str, raw_text=raw_text[:2000])
    try:
        response = await _get_llm().ainvoke([
            SystemMessage(content="You are a precise B2B analyst. Return only valid JSON."),
            HumanMessage(content=prompt),
        ])
        match = re.search(r'\{.*?\}', response.content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("industry", ""), data.get("matches", True)
    except Exception as e:
        logger.warning("Industry inference failed: %s", e)
    return "", False


# ---------------------------------------------------------------------------
# Phase 0 — Indian business directories (IndiaMart → JustDial)
# Tried first for all queries; returns [] gracefully for non-Indian locations.
# ---------------------------------------------------------------------------

async def _try_indian_directories(industry: str, location: str, max_names: int = 8) -> list[str]:
    """
    Tries IndiaMart (Phase 0A, static HTML) then JustDial (Phase 0B, JS-rendered)
    for company name discovery. Returns on the first that yields results.
    Designed to return [] silently when the location has no IndiaMart/JustDial coverage.
    """
    if not industry or not location:
        return []

    # Phase 0A — IndiaMart
    try:
        from tools.indiamart_tool import search_indiamart_companies
        results = await search_indiamart_companies.ainvoke({
            "city"       : location,
            "industry"   : industry,
            "max_results": max_names,
        })
        if results:
            names = [r["name"] for r in results if r.get("name")]
            if names:
                logger.info("company_search: Phase 0A (IndiaMart) → %d names", len(names))
                return names
    except Exception as e:
        logger.warning("company_search: Phase 0A (IndiaMart) failed — %s", e)

    # Phase 0B — JustDial fallback
    try:
        from tools.justdial_tool import search_justdial_businesses
        results = await search_justdial_businesses.ainvoke({
            "city"       : location,
            "category"   : industry,
            "max_results": max_names,
        })
        if results:
            names = [r["name"] for r in results if r.get("name")]
            if names:
                logger.info("company_search: Phase 0B (JustDial) → %d names", len(names))
                return names
    except Exception as e:
        logger.warning("company_search: Phase 0B (JustDial) failed — %s", e)

    return []


# ---------------------------------------------------------------------------
# Phase 1A — Direct company directory queries (reliable, SearXNG-independent)
# ---------------------------------------------------------------------------

_LOCATION_SLUG_MAP = {
    # India
    "bangalore": "bengaluru", "bengaluru": "bengaluru",
    "mumbai": "mumbai", "delhi": "new-delhi", "new delhi": "new-delhi",
    "hyderabad": "hyderabad", "pune": "pune", "chennai": "chennai",
    "kolkata": "kolkata", "ahmedabad": "ahmedabad", "jaipur": "jaipur",
    "kochi": "kochi", "lucknow": "lucknow", "indore": "indore",
    "india": "india",
    # Europe
    "germany": "germany", "deutschland": "germany",
    "france": "france", "netherlands": "netherlands", "spain": "spain",
    "italy": "italy", "sweden": "sweden", "norway": "norway",
    "denmark": "denmark", "finland": "finland", "austria": "austria",
    "switzerland": "switzerland", "belgium": "belgium", "poland": "poland",
    "portugal": "portugal", "czech republic": "czech-republic", "czechia": "czech-republic",
    "romania": "romania", "hungary": "hungary", "greece": "greece",
    # English-speaking
    "usa": "united-states", "us": "united-states", "united states": "united-states",
    "uk": "united-kingdom", "united kingdom": "united-kingdom",
    "australia": "australia", "canada": "canada", "new zealand": "new-zealand",
    "ireland": "ireland",
    # Asia-Pacific
    "singapore": "singapore", "japan": "japan",
    "south korea": "south-korea", "korea": "south-korea",
    "china": "china", "hong kong": "hong-kong",
    "taiwan": "taiwan", "indonesia": "indonesia", "malaysia": "malaysia",
    "thailand": "thailand", "vietnam": "vietnam", "philippines": "philippines",
    "bangladesh": "bangladesh", "sri lanka": "sri-lanka",
    # Middle East & Africa
    "uae": "united-arab-emirates", "dubai": "dubai", "abu dhabi": "abu-dhabi",
    "saudi arabia": "saudi-arabia", "israel": "israel",
    "egypt": "egypt", "south africa": "south-africa", "nigeria": "nigeria", "kenya": "kenya",
    # Americas
    "brazil": "brazil", "mexico": "mexico", "colombia": "colombia",
    "argentina": "argentina", "chile": "chile", "peru": "peru",
}


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _build_directory_urls(industry: str, location: str) -> list[str]:
    """Build direct URLs for known company directories that have predictable URL patterns."""
    urls: list[str] = []
    if not industry or not location:
        return urls
    loc_lower = location.lower()
    ind_slug = _slugify(industry)
    loc_slug = _LOCATION_SLUG_MAP.get(loc_lower, _slugify(location))
    # ensun.io: well-indexed company directory
    urls.append(f"https://ensun.io/search/{ind_slug}/{loc_slug}")
    # Europages: strong for European companies
    if loc_lower in ("germany", "france", "netherlands", "uk", "united-kingdom", "spain", "italy"):
        urls.append(f"https://www.europages.co.uk/companies/{loc_slug}/{ind_slug}.html")
    return urls


async def _try_directory_extraction(industry: str, location: str, max_names: int = 8) -> list[str]:
    """
    Tries known company directories directly using URL templates,
    bypassing SearXNG entirely. Returns company names or [].
    """
    for url in _build_directory_urls(industry, location):
        try:
            logger.info("company_search: Phase 1A trying directory URL: %s", url)
            scraped = await asyncio.wait_for(
                scrape_company_website.ainvoke({"url": url, "company_name": ""}),
                timeout=20.0,
            )
            raw_text = scraped.get("raw_text", "") or ""
            if len(raw_text) < 100:
                logger.debug("company_search: Phase 1A — too little content from %s", url)
                continue
            prompt = _EXTRACT_NAMES_PROMPT.format(text=raw_text[:3000])
            response = await _get_heavy_llm().ainvoke([HumanMessage(content=prompt)])
            arr_match = re.search(r'\[.*?\]', response.content, re.DOTALL)
            if arr_match:
                names = json.loads(arr_match.group())
                if isinstance(names, list) and names:
                    logger.info("company_search: Phase 1A extracted %d names from %s", len(names), url)
                    return [n.strip() for n in names if isinstance(n, str) and len(n.strip()) > 1][:max_names]
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning("company_search: Phase 1A failed for %s — %s", url, e)
    return []


# ---------------------------------------------------------------------------
# Phase 1B — SearXNG fallback for company name discovery
# ---------------------------------------------------------------------------

# Common English words that appear in many URLs but carry no topic signal
_URL_GENERIC_TERMS = {
    "software", "companies", "company", "vendors", "vendor", "official",
    "site", "list", "top", "best", "technology", "technologies",
    "solutions", "solution", "services", "service", "business", "global",
    "management", "systems", "system", "products", "product",
}


def _is_phase1_url_relevant(url: str, search_terms: list[str]) -> bool:
    """
    Returns True if a URL is worth scraping in Phase 1.
    Strategy: the URL must contain at least one SPECIFIC keyword from the search —
    specifically the industry root or the location — not generic words like 'software', 'companies'.
    Also blocks known time-waster domains.
    """
    if not url.startswith("http"):
        return False
    url_lower = url.lower()
    try:
        domain = url.split("/")[2].lower()
    except IndexError:
        return False
    bare = domain[4:] if domain.startswith("www.") else domain
    if any(bad in bare for bad in _PHASE1_SKIP_DOMAINS):
        return False
    # Only use non-generic, meaningful terms for URL relevance
    specific = [t.lower() for t in search_terms if len(t) >= 3 and t.lower() not in _URL_GENERIC_TERMS]
    if not specific:
        return True  # no specific terms → accept anything that isn't blocklisted
    return any(kw in url_lower for kw in specific)


async def _extract_company_names_from_web(
    list_query: str,
    max_names : int = 8,
    location  : str | None = None,
) -> list[str]:
    """
    Searches for a list-article about companies, scrapes it, and extracts company names.
    Uses relevance filter: checks URL AND title for specific keywords before scraping.
    Tries up to 8 results before giving up.
    When location is provided, passes it through so SearXNG uses categories=map,
    returning business listings instead of news articles.
    """
    search_terms = list_query.split()
    results = await searxng_search.ainvoke({
        "keywords"   : search_terms,
        "max_results": 8,
        "location"   : location,
    })
    for r in results:
        if "error" in r:
            continue
        url = r.get("url", "")
        title = r.get("title", "") or ""
        # Check URL AND title together — title is often more informative than URL
        combined_text = f"{url} {title}".lower()
        specific = [t.lower() for t in search_terms if len(t) >= 3 and t.lower() not in _URL_GENERIC_TERMS]
        if specific and not any(kw in combined_text for kw in specific):
            logger.debug("company_search: Phase 1 skipping irrelevant result: %s", url)
            continue
        try:
            domain = url.split("/")[2].lower()
        except IndexError:
            continue
        bare = domain[4:] if domain.startswith("www.") else domain
        if any(bad in bare for bad in _PHASE1_SKIP_DOMAINS):
            continue
        try:
            scraped = await scrape_company_website.ainvoke({"url": url, "company_name": ""})
            raw_text = scraped.get("raw_text", "") or ""
            if len(raw_text) < 200:
                continue
            prompt = _EXTRACT_NAMES_PROMPT.format(text=raw_text[:3000])
            response = await _get_heavy_llm().ainvoke([HumanMessage(content=prompt)])
            arr_match = re.search(r'\[.*?\]', response.content, re.DOTALL)
            if arr_match:
                names = json.loads(arr_match.group())
                if isinstance(names, list) and names:
                    logger.info("company_search: extracted %d names from %s", len(names), url)
                    return [n.strip() for n in names if isinstance(n, str) and len(n.strip()) > 1][:max_names]
        except Exception as e:
            logger.warning("company_search: name extraction failed for %s — %s", url, e)
    return []


# ---------------------------------------------------------------------------
# Phase 2 — Find actual company homepage for a given name
# ---------------------------------------------------------------------------

_COMPANY_SUFFIXES = re.compile(
    r"\b(AG|GmbH|Inc\.?|LLC|Ltd\.?|Corp\.?|Software|Solutions|Technologies|"
    r"Digital Industries|Industries|International|Group|Systems)\b",
    re.IGNORECASE,
)

def _candidate_domains(company_name: str, location: str) -> list[str]:
    """Build likely homepage URLs from the company name and location."""
    # Strip legal suffixes to get the brand core
    name = _COMPANY_SUFFIXES.sub("", company_name).strip()
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug or len(slug) < 2:
        return []
    _COUNTRY_TLDS: dict[str, list[str]] = {
        "germany": [".de", ".com"], "deutschland": [".de", ".com"],
        "uk": [".co.uk", ".com"], "united kingdom": [".co.uk", ".com"],
        "united-kingdom": [".co.uk", ".com"],
        "france": [".fr", ".com"],
        "netherlands": [".nl", ".com"],
        "australia": [".com.au", ".com"],
        "new zealand": [".co.nz", ".com"],
        "japan": [".co.jp", ".jp", ".com"],
        "brazil": [".com.br", ".com"],
        "mexico": [".com.mx", ".com"],
        "south korea": [".co.kr", ".kr", ".com"], "korea": [".co.kr", ".kr", ".com"],
        "india": [".in", ".com"],
        "bangalore": [".in", ".com"], "bengaluru": [".in", ".com"],
        "mumbai": [".in", ".com"], "delhi": [".in", ".com"],
        "hyderabad": [".in", ".com"], "pune": [".in", ".com"], "chennai": [".in", ".com"],
        "kolkata": [".in", ".com"], "ahmedabad": [".in", ".com"],
        "singapore": [".com.sg", ".sg", ".com"],
        "spain": [".es", ".com"],
        "italy": [".it", ".com"],
        "sweden": [".se", ".com"],
        "norway": [".no", ".com"],
        "denmark": [".dk", ".com"],
        "finland": [".fi", ".com"],
        "austria": [".at", ".com"],
        "switzerland": [".ch", ".com"],
        "belgium": [".be", ".com"],
        "poland": [".pl", ".com"],
        "portugal": [".pt", ".com"],
        "china": [".cn", ".com"],
        "hong kong": [".hk", ".com"],
        "indonesia": [".co.id", ".com"],
        "malaysia": [".com.my", ".my", ".com"],
        "thailand": [".co.th", ".th", ".com"],
        "philippines": [".com.ph", ".ph", ".com"],
        "israel": [".co.il", ".il", ".com"],
        "canada": [".ca", ".com"],
        "south africa": [".co.za", ".za", ".com"],
        "uae": [".ae", ".com"], "dubai": [".ae", ".com"],
        "saudi arabia": [".com.sa", ".sa", ".com"],
        "argentina": [".com.ar", ".com"],
        "colombia": [".com.co", ".com"],
        "chile": [".cl", ".com"],
        "nigeria": [".com.ng", ".ng", ".com"],
        "kenya": [".co.ke", ".ke", ".com"],
    }
    tlds = _COUNTRY_TLDS.get(location.lower(), [".com"])
    candidates: list[str] = []
    for tld in tlds:
        candidates.append(f"https://www.{slug}{tld}")
        candidates.append(f"https://{slug}{tld}")
    return candidates


async def _try_direct_domain(
    company_name: str,
    location: str,
    criteria_str: str,
    seen_roots: set,
) -> dict | None:
    """
    Before using SearXNG, try direct URL construction: {slug}.de / {slug}.com.
    Much more reliable for well-known companies with predictable domains.
    """
    for url in _candidate_domains(company_name, location):
        domain = url.split("/")[2] if url.startswith("http") else ""
        root = _root_domain(domain)
        if root in seen_roots:
            continue
        if not _is_company_url(url, seen_roots):
            continue
        try:
            scraped = await asyncio.wait_for(
                scrape_company_website.ainvoke({"url": url, "company_name": company_name}),
                timeout=12.0,
            )
            if scraped.get("confidence", 0) == 0.0:
                continue
            raw_text = scraped.pop("raw_text", "")
            industry, matches = await _infer_industry(raw_text, criteria_str)
            if not matches:
                continue
            seen_roots.add(root)
            clean_name = _clean_company_name(domain, company_name)
            scraped["name"] = clean_name
            scraped["industry"] = industry
            scraped.pop("employee_count", None)
            scraped.pop("error", None)
            logger.info("company_search: direct-domain hit '%s' (%s)", clean_name, url)
            return scraped
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug("company_search: direct domain %s failed — %s", url, e)
    return None


def _build_phase2_query(company_name: str, criteria_str: str) -> str:
    """Builds a focused SearXNG query for Phase 2 that includes industry/location context."""
    parts = [company_name]
    for line in criteria_str.splitlines():
        if line.lower().startswith("location:"):
            loc = line.split(":", 1)[1].strip()
            if loc:
                parts.append(loc)
        elif line.lower().startswith("industry:"):
            ind = line.split(":", 1)[1].strip()
            if ind:
                parts.append(ind)
    parts.append("official site")
    return " ".join(parts)


async def _find_homepage(
    company_name: str,
    criteria_str: str,
    seen_roots: set,
    skip_location_check: bool = False,
) -> dict | None:
    """
    Finds the verified homepage for a company name.
    Strategy: try direct domain construction first (fast, reliable), then fall back to SearXNG.
    skip_location_check: bypass soft location filter for Phase 1A-sourced names.
    """
    # Phase 2A — direct domain construction (fast, no SearXNG needed)
    location = ""
    for line in criteria_str.splitlines():
        if line.lower().startswith("location:"):
            location = line.split(":", 1)[1].strip()
            break
    direct = await _try_direct_domain(company_name, location, criteria_str, seen_roots)
    if direct:
        return direct

    # Phase 2B — SearXNG fallback
    query = _build_phase2_query(company_name, criteria_str)
    logger.debug("company_search: Phase 2B SearXNG for '%s': '%s'", company_name, query)
    results = await searxng_search.ainvoke({"keywords": query.split(), "max_results": 5})
    location_terms = [] if skip_location_check else _extract_location_terms(criteria_str)

    for r in results:
        if "error" in r:
            continue
        url = r.get("url", "")
        domain = url.split("/")[2] if url.startswith("http") else ""
        root = _root_domain(domain)

        if not _is_company_url(url, seen_roots):
            continue

        # Reject reseller/partner pages: company name in URL path but not in domain
        # e.g. cadopt.com/xplm/ for company "XPLM" → reseller page, not company homepage
        url_path_raw = url.split("/", 3)[3] if url.count("/") >= 3 else ""
        name_words = [w.lower() for w in company_name.split() if len(w) >= 3]
        if name_words:
            domain_has_name = any(w in root.lower() for w in name_words)
            path_has_name = bool(url_path_raw) and any(w in url_path_raw.lower() for w in name_words)
            if path_has_name and not domain_has_name:
                logger.debug("company_search: '%s' — name in path not domain, likely reseller", url)
                continue

        try:
            clean_name = _clean_company_name(domain, r.get("title", domain))
            scraped = await scrape_company_website.ainvoke({"url": url, "company_name": clean_name})
            if scraped.get("confidence", 0) == 0.0:
                continue

            raw_text = scraped.pop("raw_text", "")

            # Soft location check (skipped when Phase 1A sourced the name from a location directory)
            if location_terms:
                raw_lower = raw_text.lower()
                if not any(t in raw_lower for t in location_terms):
                    logger.debug("company_search: '%s' — no location mention, skipping", domain)
                    continue

            industry, matches = await _infer_industry(raw_text, criteria_str)
            if not matches:
                continue

            seen_roots.add(root)
            scraped["name"] = clean_name
            scraped["industry"] = industry
            scraped.pop("employee_count", None)
            scraped.pop("error", None)
            logger.info("company_search: verified company '%s' (%s)", clean_name, url)
            return scraped

        except Exception as e:
            logger.warning("company_search: homepage find failed for '%s' — %s", company_name, e)

    return None


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

async def company_search(state: GraphState) -> dict:
    """
    Company Search Agent — two-phase strategy:
      Phase 1: Searches for a list article about companies in the target space,
               extracts company names using the LLM.
      Phase 2: For each extracted name, searches directly for the company's homepage,
               verifies it with Crawl4AI, checks location presence, and infers industry.
    Short-circuit: if query_plan.company_named_directly=True, uses Apollo to look up
    the named company directly and skips the full discovery flow.
    Writes list[CompanyData] to GraphState. Never raises.
    """
    errors = list(state.get("errors", []))
    original_query = state.get("query", "")
    query_plan = state.get("query_plan", {})

    # ── Named-company short-circuit (Rule 10) ────────────────────────────────
    if query_plan.get("company_named_directly") and query_plan.get("named_company"):
        from tools.apollo_tool import search_apollo_company
        named = query_plan["named_company"]
        logger.info("company_search: named-company short-circuit for '%s'", named)
        try:
            apollo_result = await search_apollo_company.ainvoke({"company_name": named})
        except Exception as e:
            apollo_result = {}
            errors.append(f"company_search: Apollo lookup failed for '{named}' — {e}")

        if apollo_result:
            return {"companies": [apollo_result], "errors": errors}

        # Apollo found nothing — fall back to direct domain construction
        try:
            criteria_str = f"company: {named}"
            result = await asyncio.wait_for(
                _find_homepage(named, criteria_str, set(), skip_location_check=True),
                timeout=30.0,
            )
            companies = [result] if result else []
        except Exception as e:
            companies = []
            errors.append(f"company_search: homepage fallback failed for '{named}' — {e}")

        if not companies:
            errors.append(f"company_search: could not resolve '{named}' — returning stub.")
            companies = [{
                "name": named, "website": "", "industry": "",
                "revenue": "", "confidence": 0.5, "tech_stack": [], "source": "stub",
            }]
        return {"companies": companies, "errors": errors}
    # ── End short-circuit ────────────────────────────────────────────────────
    company_filters = query_plan.get("company_filters") or {}
    criteria_str = _build_criteria_str(company_filters)

    logger.info("company_search: company_filters=%s", company_filters)
    industry  = company_filters.get("industry", "") or ""
    location  = company_filters.get("location", "") or ""
    keywords  = " ".join(company_filters.get("keywords", []) or [])

    # If query_parser returned no filters, log it — the raw query will drive SearXNG directly
    if not industry and not location:
        logger.warning(
            "company_search: query_parser returned empty company_filters for '%s' "
            "— will search using raw query terms",
            original_query,
        )

    # ── Phase 0: Indian directories + OpenStreetMap Overpass (concurrent) ───
    # Both run in parallel; results are merged so each source adds unique names.
    # Overpass fires only when a location is present (physical business data).
    company_names: list[str] = []
    from_directory = False

    _osm_keywords = (company_filters.get("keywords") or [])[:2]
    if industry:
        _osm_keywords = [industry] + _osm_keywords

    async def _fetch_overpass_names() -> list[str]:
        if not location:
            return []
        raw = await search_overpass_businesses.ainvoke({
            "keywords"   : _osm_keywords,
            "location"   : location,
            "max_results": 8,
        })
        return [r["name"] for r in raw if r.get("name")]

    _p0_dir, _p0_osm = await asyncio.gather(
        asyncio.wait_for(_try_indian_directories(industry, location, max_names=8), timeout=45.0),
        asyncio.wait_for(_fetch_overpass_names(), timeout=35.0),
        return_exceptions=True,
    )

    if isinstance(_p0_dir, BaseException):
        errors.append(f"company_search: Phase 0 (IndiaMart/JustDial) failed — {_p0_dir}")
        _p0_dir = []
    if isinstance(_p0_osm, BaseException):
        errors.append(f"company_search: Phase 0 (Overpass) failed — {_p0_osm}")
        _p0_osm = []

    _seen_p0: set[str] = set()
    for _n in (_p0_dir or []) + (_p0_osm or []):
        if _n.lower() not in _seen_p0:
            _seen_p0.add(_n.lower())
            company_names.append(_n)

    if company_names:
        from_directory = True
        logger.info(
            "company_search: Phase 0 — %d directory + %d OSM = %d unique names.",
            len(_p0_dir or []), len(_p0_osm or []), len(company_names),
        )

    # ── Phase 1A: try known company directories directly (most reliable) ────────
    if not company_names:
        try:
            company_names = await asyncio.wait_for(
                _try_directory_extraction(industry, location, max_names=8),
                timeout=60.0,
            )
            if company_names:
                from_directory = True
        except Exception as e:
            errors.append(f"company_search: Phase 1A failed — {e}")

    # ── Phase 1B: SearXNG fallback — only if Phase 1A found nothing ─────────
    if not company_names:
        def _list_queries() -> list[str]:
            ind = industry or ""
            loc = location or ""
            queries: list[str] = []
            if ind and loc:
                queries.append(f"{ind} companies {loc}")
                queries.append(f"top {ind} companies {loc}")
                queries.append(f"{ind} B2B companies {loc}")
            elif ind:
                queries.append(f"top {ind} B2B companies")
                queries.append(f"{ind} vendors list")
            elif loc:
                # No industry known — search using the raw query so we don't
                # silently substitute "software companies" for whatever the user meant
                queries.append(f"B2B companies {loc}")
                queries.append(original_query)
            else:
                # No industry and no location — use the raw query verbatim
                queries.append(original_query)
            return queries

        for list_query in _list_queries():
            logger.info("company_search: Phase 1B SearXNG query: '%s'", list_query)
            try:
                names = await asyncio.wait_for(
                    _extract_company_names_from_web(
                        list_query,
                        max_names=8,
                        location=location or None,
                    ),
                    timeout=60.0,
                )
                if names:
                    company_names = names
                    break
            except asyncio.TimeoutError:
                errors.append(f"company_search: Phase 1B timed out for query '{list_query}'")
            except Exception as e:
                errors.append(f"company_search: Phase 1B failed for '{list_query}' — {e}")

    logger.info("company_search: Phase 1 extracted %d names: %s", len(company_names), company_names)

    # ── Phase 2: verify each company name → find homepage ───────────────────
    seen_roots: set[str] = set()
    all_companies: list[dict] = []

    # Filter out generic/single-word names that aren't real companies
    _GENERIC_NAMES = _GENERIC_DOMAIN_WORDS | {
        "fintech", "startup", "company", "companies", "vendor", "vendors",
        "india", "bangalore", "bengaluru", "germany", "usa", "uk", "singapore",
        "plm", "erp", "crm", "saas", "cloud", "top", "best", "list",
        "provider", "providers", "solution", "solutions",
    }
    company_names = [
        n for n in company_names
        if len(n.split()) >= 1 and n.lower().strip() not in _GENERIC_NAMES and len(n) >= 3
    ]

    if company_names:
        _HOMEPAGE_TIMEOUT = 35.0   # per company — prevents indefinite hang on slow URLs

        async def _safe_find_homepage(name: str) -> dict | None:
            try:
                return await asyncio.wait_for(
                    _find_homepage(name, criteria_str, seen_roots, skip_location_check=from_directory),
                    timeout=_HOMEPAGE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("company_search: timeout finding homepage for '%s'", name)
                return None
            except Exception as e:
                logger.warning("company_search: error finding homepage for '%s' — %s", name, e)
                return None

        # Sequential — not parallel — to avoid hammering Groq TPM rate limit
        # (4 parallel industry inference calls at ~1669 tokens each exceeds 6000 TPM)
        for name in company_names:
            r = await _safe_find_homepage(name)
            if r is not None:
                all_companies.append(r)
    else:
        errors.append("company_search: Phase 1 returned no valid names — no companies found.")

    # Sort by confidence and cap at 5 to conserve token quota
    all_companies.sort(key=lambda c: c.get("confidence", 0.0), reverse=True)
    all_companies = all_companies[:5]

    logger.info("company_search: found %d companies.", len(all_companies))

    return {
        "companies": all_companies,
        "errors"   : errors,
        "status"   : "companies_found" if all_companies else "no_companies_found",
    }
