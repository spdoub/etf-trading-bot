"""Daily data collection from diverse free sources for sentiment analysis.

Five source categories, each returning structured DataItem records:
    1. Financial headlines     — Reuters, AP, MarketWatch, CNBC
    2. Local / regional US     — BizJournals + newspapers across 15 cities
    3. Government contracts    — USASpending.gov (free) + SAM.gov (optional key)
    4. Job posting trends      — Indeed RSS, Remotive, USAJobs (optional key)
    5. Foreign financial news  — Nikkei Asia, Deutsche Welle, Korea Herald
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# Optional keys — sources degrade gracefully when absent
SAM_GOV_API_KEY = os.getenv("SAM_GOV_API_KEY")
USAJOBS_API_KEY = os.getenv("USAJOBS_API_KEY")
USAJOBS_EMAIL = os.getenv("USAJOBS_EMAIL")

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "etf-trading-bot/1.0 (research)"})
REQUEST_TIMEOUT = 15


# ═══════════════════════════════════════════════════════════════════════════
# DataItem — the universal return type for every source
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DataItem:
    """Single piece of collected intelligence."""
    headline: str
    summary: str
    source: str       # e.g. "reuters", "bizjournals_chicago", "usaspending"
    category: str     # e.g. "financial_headlines", "local_us_news", …
    timestamp: str    # ISO 8601

    def as_text(self) -> str:
        """One-liner ready for LLM consumption."""
        ts_short = self.timestamp[:16]
        parts = [f"[{self.source} | {ts_short}]", self.headline]
        if self.summary:
            parts.append(self.summary[:300])
        return " — ".join(parts)

    def as_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ts(struct_time) -> str:
    """Convert feedparser time struct to ISO string."""
    if struct_time is None:
        return datetime.now(timezone.utc).isoformat()
    try:
        return datetime(*struct_time[:6], tzinfo=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _strip_html(raw: str, max_len: int = 500) -> str:
    if not raw:
        return ""
    return BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)[:max_len]


def _parse_rss(url: str, source_label: str, category: str,
               max_items: int = 15) -> list[DataItem]:
    """Generic RSS/Atom → DataItem parser.  Never raises."""
    items: list[DataItem] = []
    try:
        feed = feedparser.parse(url)
        if feed.bozo and not feed.entries:
            log.debug("RSS bozo with no entries [%s]: %s", source_label, feed.bozo_exception)
            return items
        for entry in feed.entries[:max_items]:
            headline = (entry.get("title") or "").strip()
            if not headline:
                continue
            summary = _strip_html(
                entry.get("summary") or entry.get("description") or ""
            )
            ts = _ts(entry.get("published_parsed") or entry.get("updated_parsed"))
            items.append(DataItem(
                headline=headline,
                summary=summary,
                source=source_label,
                category=category,
                timestamp=ts,
            ))
    except Exception as exc:
        log.warning("RSS [%s] %s — %s", source_label, url, exc)
    return items


# ═══════════════════════════════════════════════════════════════════════════
# 1.  TOP FINANCIAL HEADLINES
# ═══════════════════════════════════════════════════════════════════════════

FINANCIAL_FEEDS: dict[str, str] = {
    "reuters": (
        "https://www.reutersagency.com/feed/"
        "?best-topics=business-finance&post_type=best"
    ),
    "ap_business": "https://rsshub.app/apnews/topics/business",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "marketwatch_markets": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "cnbc_top": (
        "https://search.cnbc.com/rs/search/combinedcms/view.xml"
        "?partnerId=wrss01&id=100003114"
    ),
    "cnbc_finance": (
        "https://search.cnbc.com/rs/search/combinedcms/view.xml"
        "?partnerId=wrss01&id=10000664"
    ),
}


def fetch_financial_headlines(max_per_feed: int = 15) -> list[DataItem]:
    """Top financial headlines from major wire services and outlets."""
    items: list[DataItem] = []
    for label, url in FINANCIAL_FEEDS.items():
        items.extend(_parse_rss(url, label, "financial_headlines", max_per_feed))
    log.info("Financial headlines: %d items from %d feeds",
             len(items), len(FINANCIAL_FEEDS))
    return items


# ═══════════════════════════════════════════════════════════════════════════
# 2.  LOCAL / REGIONAL US BUSINESS NEWS
# ═══════════════════════════════════════════════════════════════════════════

LOCAL_US_FEEDS: dict[str, str] = {
    # BizJournals — reliable business RSS across major metros
    "bizj_atlanta": "https://feeds.bizjournals.com/bizj_atlanta",
    "bizj_boston": "https://feeds.bizjournals.com/bizj_boston",
    "bizj_chicago": "https://feeds.bizjournals.com/bizj_chicago",
    "bizj_dallas": "https://feeds.bizjournals.com/bizj_dallas",
    "bizj_denver": "https://feeds.bizjournals.com/bizj_denver",
    "bizj_detroit": "https://feeds.bizjournals.com/bizj_detroit",
    "bizj_houston": "https://feeds.bizjournals.com/bizj_houston",
    "bizj_minneapolis": "https://feeds.bizjournals.com/bizj_twincities",
    "bizj_miami": "https://feeds.bizjournals.com/bizj_southflorida",
    "bizj_phoenix": "https://feeds.bizjournals.com/bizj_phoenix",
    "bizj_pittsburgh": "https://feeds.bizjournals.com/bizj_pittsburgh",
    "bizj_portland": "https://feeds.bizjournals.com/bizj_portland",
    "bizj_sanfrancisco": "https://feeds.bizjournals.com/bizj_sanfrancisco",
    "bizj_seattle": "https://feeds.bizjournals.com/bizj_seattle",
    "bizj_dc": "https://feeds.bizjournals.com/bizj_washington",
    # Direct newspaper business sections
    "la_times_biz": "https://www.latimes.com/business/rss2.0.xml",
    "seattle_times_biz": "https://www.seattletimes.com/business/feed/",
    "denver_post_biz": "https://www.denverpost.com/business/feed/",
}


def fetch_local_us_news(max_per_feed: int = 10) -> list[DataItem]:
    """Economic and business stories from 15+ US metro areas."""
    items: list[DataItem] = []
    for label, url in LOCAL_US_FEEDS.items():
        items.extend(_parse_rss(url, label, "local_us_news", max_per_feed))
    log.info("Local US news: %d items from %d feeds",
             len(items), len(LOCAL_US_FEEDS))
    return items


# ═══════════════════════════════════════════════════════════════════════════
# 3.  GOVERNMENT CONTRACT AWARDS
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_usaspending(days_back: int = 7, limit: int = 25) -> list[DataItem]:
    """Recent federal contract awards from USASpending.gov (no key needed)."""
    items: list[DataItem] = []
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    payload = {
        "subawards": False,
        "limit": limit,
        "page": 1,
        "sort": "Award Amount",
        "order": "desc",
        "filters": {
            "time_period": [{
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
            }],
            "award_type_codes": ["A", "B", "C", "D"],
        },
        "fields": [
            "Award ID",
            "Recipient Name",
            "Award Amount",
            "Awarding Agency",
            "Description",
            "Start Date",
        ],
    }
    try:
        resp = _SESSION.post(
            "https://api.usaspending.gov/api/v2/search/spending_by_award/",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        for r in resp.json().get("results", []):
            recipient = r.get("Recipient Name") or "Unknown"
            amount = r.get("Award Amount")
            agency = r.get("Awarding Agency") or "Unknown agency"
            desc = r.get("Description") or ""
            award_date = r.get("Start Date") or end.strftime("%Y-%m-%d")
            amt_str = f"${amount:,.0f}" if amount else "undisclosed"
            headline = f"{agency} awards {amt_str} contract to {recipient}"
            items.append(DataItem(
                headline=headline,
                summary=desc[:500],
                source="usaspending",
                category="government_contracts",
                timestamp=f"{award_date}T00:00:00+00:00",
            ))
    except Exception as exc:
        log.warning("USASpending fetch failed: %s", exc)
    return items


def _fetch_sam_gov(days_back: int = 7, limit: int = 20) -> list[DataItem]:
    """Contract opportunities from SAM.gov (requires free API key)."""
    if not SAM_GOV_API_KEY:
        log.debug("SAM_GOV_API_KEY not set — skipping SAM.gov")
        return []
    items: list[DataItem] = []
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    try:
        resp = _SESSION.get(
            "https://api.sam.gov/opportunities/v2/search",
            params={
                "api_key": SAM_GOV_API_KEY,
                "postedFrom": start.strftime("%m/%d/%Y"),
                "postedTo": end.strftime("%m/%d/%Y"),
                "limit": limit,
                "offset": 0,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        for opp in resp.json().get("opportunitiesData", []):
            title = opp.get("title", "").strip()
            if not title:
                continue
            agency = opp.get("fullParentPathName") or opp.get("department") or ""
            desc = opp.get("description") or ""
            posted = opp.get("postedDate") or end.strftime("%Y-%m-%d")
            items.append(DataItem(
                headline=title,
                summary=f"{agency}. {_strip_html(desc, 400)}".strip(),
                source="sam_gov",
                category="government_contracts",
                timestamp=f"{posted}T00:00:00+00:00",
            ))
    except Exception as exc:
        log.warning("SAM.gov fetch failed: %s", exc)
    return items


def fetch_government_contracts() -> list[DataItem]:
    """Aggregate recent government contract awards and opportunities."""
    items = _fetch_usaspending()
    items.extend(_fetch_sam_gov())
    log.info("Government contracts: %d items", len(items))
    return items


# ═══════════════════════════════════════════════════════════════════════════
# 4.  JOB POSTING TRENDS
# ═══════════════════════════════════════════════════════════════════════════

INDEED_RSS_QUERIES = [
    ("indeed_hiring", "https://www.indeed.com/rss?q=hiring&sort=date&limit=15"),
    ("indeed_layoffs", "https://www.indeed.com/rss?q=urgent+hiring&sort=date&limit=15"),
    ("indeed_tech", "https://www.indeed.com/rss?q=software+engineer&sort=date&limit=10"),
    ("indeed_finance", "https://www.indeed.com/rss?q=financial+analyst&sort=date&limit=10"),
]


def _fetch_indeed_rss() -> list[DataItem]:
    """Job postings via Indeed's RSS interface."""
    items: list[DataItem] = []
    for label, url in INDEED_RSS_QUERIES:
        items.extend(_parse_rss(url, label, "job_trends"))
    return items


def _fetch_remotive(limit: int = 20) -> list[DataItem]:
    """Remote job postings from Remotive (free, no key)."""
    items: list[DataItem] = []
    try:
        resp = _SESSION.get(
            "https://remotive.com/api/remote-jobs",
            params={"limit": limit},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        for job in resp.json().get("jobs", []):
            title = job.get("title", "").strip()
            if not title:
                continue
            company = job.get("company_name") or ""
            category = job.get("category") or ""
            pub = job.get("publication_date") or datetime.now(timezone.utc).isoformat()
            items.append(DataItem(
                headline=f"{title} at {company}" if company else title,
                summary=f"Category: {category}. {_strip_html(job.get('description', ''), 300)}",
                source="remotive",
                category="job_trends",
                timestamp=pub,
            ))
    except Exception as exc:
        log.warning("Remotive fetch failed: %s", exc)
    return items


def _fetch_usajobs(limit: int = 25) -> list[DataItem]:
    """Recent federal job postings from USAJobs.gov (optional key)."""
    if not USAJOBS_API_KEY or not USAJOBS_EMAIL:
        log.debug("USAJOBS credentials not set — skipping")
        return []
    items: list[DataItem] = []
    try:
        resp = _SESSION.get(
            "https://data.usajobs.gov/api/search",
            params={"ResultsPerPage": limit, "DatePosted": 7},
            headers={
                "Authorization-Key": USAJOBS_API_KEY,
                "User-Agent": USAJOBS_EMAIL,
                "Host": "data.usajobs.gov",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        results = (resp.json()
                   .get("SearchResult", {})
                   .get("SearchResultItems", []))
        for item in results:
            mp = item.get("MatchedObjectDescriptor", {})
            title = mp.get("PositionTitle", "").strip()
            if not title:
                continue
            org = mp.get("OrganizationName") or ""
            loc = mp.get("PositionLocationDisplay") or ""
            salary = mp.get("PositionRemuneration", [{}])
            sal_str = ""
            if salary:
                s = salary[0] if isinstance(salary, list) else salary
                sal_str = f"${s.get('MinimumRange', '?')}–${s.get('MaximumRange', '?')}"
            pub = mp.get("PublicationStartDate") or datetime.now(timezone.utc).isoformat()
            items.append(DataItem(
                headline=f"{title} — {org}" if org else title,
                summary=f"{loc}. {sal_str}".strip(),
                source="usajobs",
                category="job_trends",
                timestamp=pub,
            ))
    except Exception as exc:
        log.warning("USAJobs fetch failed: %s", exc)
    return items


def fetch_job_trends() -> list[DataItem]:
    """Aggregate job posting data from multiple free sources."""
    items = _fetch_indeed_rss()
    items.extend(_fetch_remotive())
    items.extend(_fetch_usajobs())
    log.info("Job trends: %d items", len(items))
    return items


# ═══════════════════════════════════════════════════════════════════════════
# 5.  FOREIGN FINANCIAL NEWS (English-language editions)
# ═══════════════════════════════════════════════════════════════════════════

FOREIGN_FEEDS: dict[str, str] = {
    # Japan
    "nikkei_asia": "https://asia.nikkei.com/rss/feed/nar",
    "japan_times_biz": "https://www.japantimes.co.jp/feed/business/",
    # Germany (Deutsche Welle English business — proxy for Handelsblatt coverage)
    "dw_business": "https://rss.dw.com/xml/rss-en-bus",
    "dw_economy": "https://rss.dw.com/xml/rss-en-eco",
    # South Korea
    "korea_herald_biz": "http://www.koreaherald.com/common/rss_xml.php?ct=102",
    "korea_herald_natl": "http://www.koreaherald.com/common/rss_xml.php?ct=101",
    "korea_joongang": "https://koreajoongangdaily.joins.com/section/rss/business",
}


def fetch_foreign_financial_news(max_per_feed: int = 12) -> list[DataItem]:
    """English-language financial news from Japanese, German, and Korean outlets."""
    items: list[DataItem] = []
    for label, url in FOREIGN_FEEDS.items():
        items.extend(_parse_rss(url, label, "foreign_financial_news", max_per_feed))
    log.info("Foreign financial news: %d items from %d feeds",
             len(items), len(FOREIGN_FEEDS))
    return items


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def collect_all() -> dict[str, list[DataItem]]:
    """Run every source and return items grouped by category.

    Returns:
        {
            "financial_headlines":    [...],
            "local_us_news":         [...],
            "government_contracts":  [...],
            "job_trends":            [...],
            "foreign_financial_news": [...],
        }
    """
    data = {
        "financial_headlines": fetch_financial_headlines(),
        "local_us_news": fetch_local_us_news(),
        "government_contracts": fetch_government_contracts(),
        "job_trends": fetch_job_trends(),
        "foreign_financial_news": fetch_foreign_financial_news(),
    }
    total = sum(len(v) for v in data.values())
    log.info("Data collection complete — %d total items across %d categories",
             total, len(data))
    return data


def collect_all_text() -> list[str]:
    """Convenience: flat list of LLM-ready strings from every source."""
    data = collect_all()
    lines: list[str] = []
    for category_items in data.values():
        lines.extend(item.as_text() for item in category_items)
    return lines
