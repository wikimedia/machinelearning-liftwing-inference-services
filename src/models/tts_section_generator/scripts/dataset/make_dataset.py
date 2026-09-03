#!/usr/bin/env python3

import csv
import datetime
import http.client
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MW_API = "https://en.wikipedia.org/w/api.php"
LIFTWING = "https://inference.svc.eqiad.wmnet:30443/v1/models"

# internal isvc hostnames, from operations/deployment-charts helmfile.d/ml-services
LW_HOST = {
    "articlequality": "articlequality.article-models.wikimedia.org",
    "outlink-topic-model": "outlink-topic-model.articletopic-outlink.wikimedia.org",
}
UA = "WMF-ML-Team-T432128-dataset/1.0 (bwojtowicz@wikimedia.org)"

SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", 10000))
SEEDS_PER_TOPIC = 50
RECS_PER_SEED = 3
TOPIC_TARGET = SEEDS_PER_TOPIC * (1 + RECS_PER_SEED)  # articles per topic
MIN_LANGLINKS = 2  # relaxed from 10 for topic coverage (T432128, 2026-08-10)
MIN_QUALITY = 0.70

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dataset.csv")

# https://github.com/censor-text/profanity-list list/en.txt (T432128 decision)
with open(os.path.join(HERE, "keywords_en.txt")) as _fh:
    KEYWORDS = [line.strip() for line in _fh if line.strip()]
KEYWORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in KEYWORDS) + r")\b", re.IGNORECASE
)

# list pages make poor audio content - dropped from seeds AND recommendations:
# title prefix as cheap backstop + WikiProject assessment classes
LIST_RE = re.compile(r"^Lists? of ")
LIST_CLASSES = {"List", "FL", "SIA", "Disambig"}

NOW = datetime.datetime.now(datetime.timezone.utc)
EDITED_CUTOFF = (NOW - datetime.timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
WEEK_AGO = (NOW - datetime.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

# ticket topic (CirrusSearch articletopic keyword) -> outlink model label
TOPICS = {
    "architecture": "Culture.Visual_arts.Architecture",
    "visual-arts": "Culture.Visual_arts.Visual_arts*",
    "comics-and-anime": "Culture.Visual_arts.Comics_and_Anime",
    "entertainment": "Culture.Media.Entertainment",
    "fashion": "Culture.Visual_arts.Fashion",
    "books": "Culture.Media.Books",
    "music": "Culture.Media.Music",
    "performing-arts": "Culture.Performing_arts",
    "sports": "Culture.Sports",
    "films": "Culture.Media.Films",
    "video-games": "Culture.Media.Video_games",
    "biography": "Culture.Biography.Biography*",
    "women": "Culture.Biography.Women",
    "business-and-economics": "History_and_Society.Business_and_economics",
    "education": "History_and_Society.Education",
    "food-and-drink": "Culture.Food_and_drink",
    "history": "History_and_Society.History",
    "military-and-warfare": "History_and_Society.Military_and_warfare",
    "philosophy-and-religion": "Culture.Philosophy_and_religion",
    "politics-and-government": "History_and_Society.Politics_and_government",
    "society": "History_and_Society.Society",
    "transportation": "History_and_Society.Transportation",
    "biology": "STEM.Biology",
    "chemistry": "STEM.Chemistry",
    "internet-culture": "Culture.Internet_culture",
    "geographical": "Geography.Geographical",
    "engineering": "STEM.Engineering",
    "stem": "STEM.STEM*",
    "mathematics": "STEM.Mathematics",
    "medicine-and-health": "STEM.Medicine_&_Health",
    "physics": "STEM.Physics",
    "technology": "STEM.Technology",
    "africa": "Geography.Regions.Africa.Africa*",
    "asia": "Geography.Regions.Asia.Asia*",
    "central-america": "Geography.Regions.Americas.Central_America",
    "europe": "Geography.Regions.Europe.Europe*",
    "north-america": "Geography.Regions.Americas.North_America",
    "oceania": "Geography.Regions.Oceania",
    "south-america": "Geography.Regions.Americas.South_America",
}


def http_json(url, payload=None, headers=None):
    headers = {"User-Agent": UA, "Content-Type": "application/json", **(headers or {})}
    data = json.dumps(payload).encode() if payload is not None else None
    for attempt in (1, 2, 3):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            # 500 from the models is deterministic (unprocessable article) - fail fast
            if attempt == 3 or e.code not in (429, 502, 503, 504):
                raise
            print(f"  HTTP {e.code}, retrying in {30 * attempt}s")
            time.sleep(30 * attempt)
        except (OSError, http.client.HTTPException) as e:
            # timeouts, resets, proxy hiccups - transient over a 20h run
            if attempt == 3:
                raise
            print(f"  {type(e).__name__}: {e}, retrying in {30 * attempt}s")
            time.sleep(30 * attempt)


def mw_get(**params):
    time.sleep(0.25)  # stay polite with the MW API
    params.update(format="json", formatversion=2)
    # MW API reports errors as 200 + {"error": ...} body; retry, then degrade
    for attempt in (1, 2, 3):
        doc = http_json(f"{MW_API}?{urllib.parse.urlencode(params)}")
        if "error" not in doc:
            return doc
        print(f"  MW API error ({doc['error'].get('code')}), attempt {attempt}")
        time.sleep(10 * attempt)
    print(f"  giving up on request: {doc['error']}")
    return {}


def search(query, limit, sort="relevance"):
    titles, offset = [], 0
    while len(titles) < limit:
        doc = mw_get(
            action="query",
            list="search",
            srsearch=query,
            srnamespace=0,
            srlimit=min(500, limit - len(titles)),
            sroffset=offset,
            srsort=sort,
        )
        hits = doc.get("query", {}).get("search", [])
        titles += [h["title"] for h in hits]
        if "continue" not in doc or not hits:
            break
        offset = doc["continue"]["sroffset"]
    return list(dict.fromkeys(titles))[:limit]


def query_pages(titles, batch, **props):
    """action=query over title batches, following continuation."""
    for i in range(0, len(titles), batch):
        params = dict(action="query", titles="|".join(titles[i : i + batch]), **props)
        while True:
            doc = mw_get(**params)
            yield from doc.get("query", {}).get("pages", [])
            if "continue" not in doc:
                break
            params.update(doc["continue"])


def fetch_meta(titles):
    """title -> current rev_id, last-edit timestamp, langlinks count and
    WikiProject assessment classes."""
    meta = {t: {"revid": None, "ts": "", "ll": 0, "pa": set()} for t in titles}
    for p in query_pages(
        titles,
        50,
        prop="revisions|langlinks|pageassessments",
        rvprop="ids|timestamp",
        lllimit="max",
        palimit="max",
    ):
        if p.get("revisions"):
            meta[p["title"]]["revid"] = p["revisions"][0]["revid"]
            meta[p["title"]]["ts"] = p["revisions"][0]["timestamp"]
        meta[p["title"]]["ll"] += len(p.get("langlinks", []))
        for a in (p.get("pageassessments") or {}).values():
            meta[p["title"]]["pa"].add(a.get("class"))
    return meta


def assessed_lists(titles):
    """Subset of titles any WikiProject assesses as a list-ish class."""
    listy = set()
    for p in query_pages(titles, 50, prop="pageassessments", palimit="max"):
        classes = {a.get("class") for a in (p.get("pageassessments") or {}).values()}
        if classes & LIST_CLASSES:
            listy.add(p["title"])
    return listy


def rec_candidates(seed, used):
    """Seed's morelike results that pass the recommendation filters:
    unclaimed, not a list page, no keyword in the lead."""
    cands = [
        t
        for t in search(f"morelike:{seed}", 50)
        if t not in used and not LIST_RE.match(t)
    ]
    listy = assessed_lists(cands)
    cands = [t for t in cands if t not in listy]
    leads = fetch_leads(cands)
    return [t for t in cands if not KEYWORD_RE.search(leads[t])]


def edit_rate(title):
    """Edits/day over the last 7 days (capped at 500 revisions)."""
    doc = mw_get(
        action="query",
        prop="revisions",
        titles=title,
        rvprop="timestamp",
        rvlimit=500,
        rvend=WEEK_AGO,
    )
    pages = doc.get("query", {}).get("pages") or [{}]
    return round(len(pages[0].get("revisions", [])) / 7, 2)


def fetch_pageviews(titles):
    """title -> total views over the last 60 days."""
    views = dict.fromkeys(titles, 0)
    for p in query_pages(titles, 50, prop="pageviews"):
        views[p["title"]] += sum(v for v in (p.get("pageviews") or {}).values() if v)
    return views


def fetch_leads(titles):
    """title -> plain-text lead section."""
    leads = dict.fromkeys(titles, "")
    for p in query_pages(
        titles, 20, prop="extracts", exintro=1, explaintext=1, exlimit="max"
    ):
        leads[p["title"]] += p.get("extract") or ""
    return leads


def lw_predict(model, payload):
    headers = {"Host": LW_HOST[model]}
    return http_json(f"{LIFTWING}/{model}:predict", payload, headers)


def quality_score(revid):
    return lw_predict("articlequality", {"rev_id": revid, "lang": "en"})["score"]


def topic_score(title, label):
    doc = lw_predict(
        "outlink-topic-model",
        {"page_title": title.replace(" ", "_"), "lang": "en", "threshold": 0.0},
    )
    scores = {r["topic"]: r["score"] for r in doc["prediction"]["results"]}
    return scores.get(label, 0.0)


def url_for(title):
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
        title.replace(" ", "_")
    )


def build_topic(topic, label, writer, used):
    fa = search(f'articletopic:{topic} incategory:"Featured articles"', 10_000)
    sample = search(f"articletopic:{topic}", SAMPLE_SIZE, sort="random")
    views = fetch_pageviews(sample)
    deciles = statistics.quantiles(views.values(), n=10)
    p50, p90 = deciles[4], deciles[8]
    band = [t for t in sample if p50 <= views[t] <= p90]
    # no duplicates across the dataset: drop articles claimed by earlier topics
    pool = [
        t for t in dict.fromkeys(fa + band) if t not in used and not LIST_RE.match(t)
    ]
    meta = fetch_meta(pool)
    pool = [
        t
        for t in pool
        if meta[t]["revid"]
        and meta[t]["ll"] >= MIN_LANGLINKS
        and meta[t]["ts"] < EDITED_CUTOFF
        and not meta[t]["pa"] & LIST_CLASSES
    ]
    after_meta = len(pool)
    leads = fetch_leads(pool)
    pool = [t for t in pool if not KEYWORD_RE.search(leads[t])]

    quality = {}
    for n, t in enumerate(pool, 1):
        try:
            quality[t] = quality_score(meta[t]["revid"])
        except urllib.error.HTTPError as e:
            print(f"  skipping {t}: articlequality HTTP {e.code}")
        if n % 25 == 0:
            print(f"  quality {n}/{len(pool)}")
    passed = [t for t in pool if quality.get(t, 0) >= MIN_QUALITY]

    scores = {}
    for n, t in enumerate(passed, 1):
        try:
            scores[t] = topic_score(t, label)
        except urllib.error.HTTPError as e:
            print(f"  skipping {t}: outlink HTTP {e.code}")
        if n % 25 == 0:
            print(f"  topic-score {n}/{len(passed)}")
    seeds = sorted(scores, key=scores.get, reverse=True)[:SEEDS_PER_TOPIC]

    print(
        f"  funnel: fa={len(fa)} sample={len(sample)} band={len(band)} "
        f"langlinks+24h+lists={after_meta} keywords={len(pool)} "
        f"quality>={MIN_QUALITY}: {len(passed)} seeds={len(seeds)}"
    )
    if len(seeds) < SEEDS_PER_TOPIC:
        print(f"  WARNING: only {len(seeds)} seeds for {topic}")

    used.update(seeds)
    pools = {}
    for n, seed in enumerate(seeds, 1):
        pools[seed] = rec_candidates(seed, used)
        if n % 10 == 0:
            print(f"  morelike {n}/{len(seeds)}")

    # topics short of 50 seeds get extra recommendations instead of a lower
    # quality bar (T432128, 2026-08-13); one per seed per round so that
    # higher-topic-score seeds take the surplus first but no seed dominates
    recs = {seed: [] for seed in seeds}
    target = TOPIC_TARGET - len(seeds)
    total = 0
    while total < target:
        before = total
        for seed in seeds:
            if total >= target:
                break
            while pools[seed]:
                cand = pools[seed].pop(0)
                if cand not in used:
                    recs[seed].append(cand)
                    used.add(cand)
                    total += 1
                    break
        if total == before:
            print(f"  WARNING: pools dry, only {total}/{target} recommendations")
            break

    for seed in seeds:
        writer.writerow([topic, seed, url_for(seed), "Seed", edit_rate(seed)])
        for rec in recs[seed]:
            writer.writerow(
                [topic, rec, url_for(rec), "Recommendation", edit_rate(rec)]
            )


def main():
    topics = sys.argv[1:] or list(TOPICS)
    done, used = set(), set()
    has_header = False
    if os.path.exists(OUT):
        with open(OUT, newline="") as fh:
            reader = csv.DictReader(fh)
            # a crashed run may have flushed the header but no rows; "no rows
            # yet" must not mean "write the header again" on restart
            has_header = reader.fieldnames is not None
            for row in reader:
                done.add(row["Topic"])
                used.add(row["Article title"])
    with open(OUT, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not has_header:
            writer.writerow(
                [
                    "Topic",
                    "Article title",
                    "Article URL",
                    "Type",
                    "Edit rate (edits/day, last 7d)",
                ]
            )
        for topic in topics:
            if topic in done:
                print(f"=== {topic}: already in {OUT}, skipping")
                continue
            print(f"=== {topic} ({TOPICS[topic]})")
            t0 = time.time()
            build_topic(topic, TOPICS[topic], writer, used)
            fh.flush()
            print(f"  done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
