#!/usr/bin/env python3
"""Regenerate wikis.tsv (wiki_id -> wikipedia subdomain) from the sitematrix API.

The second column is the *served subdomain* of {sub}.wikipedia.org, used by the
outlink topic model as the MW API Host header. It is NOT the ISO language code:
the two differ for e.g. alswiki (als, not gsw) and be_x_oldwiki (be-tarask).

In addition to canonical MediaWiki dbnames, the output includes alias rows for
common non-canonical wiki_ids observed from callers (T435586): {subdomain}wiki
and {language code}wiki forms where those differ from the dbname, e.g.
be-taraskwiki -> be-tarask (dbname be_x_oldwiki) and nbwiki -> no (dbname
nowiki). Aliases never shadow a real dbname: any candidate that collides with
a canonical dbname is skipped, so e.g. simplewiki traffic can never be
captured by an "enwiki" alias.

Norwegian is special-cased: its sitematrix group code is "no" but the wiki's
content language is "nb" (per siteinfo), and callers synthesize nbwiki from
the content language. It is the only Wikipedia where the two diverge.

Usage:
    python3 generate_wikis_tsv.py > wikis.tsv

Includes closed wikis (they still serve reads). Private/fishbowl wikis
(e.g. arbcom-en.wikipedia.org) are excluded by the code == "wiki" family
check; the response does carry private/fishbowl/closed flags if stricter
filtering is ever needed.
"""

import json
import sys
import urllib.request

SITEMATRIX_URL = (
    "https://meta.wikimedia.org/w/api.php"
    "?action=sitematrix&format=json&formatversion=2"
    "&smsiteprop=url|dbname|code&smlangprop=site|code"
)
UA = "WMF ML Team wikis.tsv generator (outlink-topic-model)"

# alias -> canonical dbname, for divergences sitematrix cannot derive.
MANUAL_ALIASES = {
    # nowiki's content language is "nb"; sitematrix group code is "no".
    "nbwiki": "nowiki",
}


def main() -> None:
    req = urllib.request.Request(SITEMATRIX_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as resp:
        sitematrix = json.load(resp)["sitematrix"]

    rows = []  # (dbname, subdomain, language group code)
    for key, group in sitematrix.items():
        if key == "count":
            continue
        sites = group if key == "specials" else group.get("site", [])
        langcode = group.get("code") if isinstance(group, dict) else None
        for site in sites:
            if site.get("code") != "wiki":  # Wikipedia family only
                continue
            host = site["url"].removeprefix("https://")
            if not host.endswith(".wikipedia.org"):
                print(f"skipping {site['dbname']}: {site['url']}", file=sys.stderr)
                continue
            rows.append((site["dbname"], host.removesuffix(".wikipedia.org"), langcode))

    dbnames = {db: sub for db, sub, _ in rows}

    aliases = {}  # alias wiki_id -> canonical dbname
    for db, sub, lang in rows:
        for candidate in (f"{sub}wiki", f"{lang}wiki" if lang else None):
            if candidate is None or candidate == db:
                continue
            if candidate in dbnames:  # never shadow a real dbname
                continue
            if aliases.get(candidate, db) != db:  # ambiguous across wikis
                print(f"skipping ambiguous alias {candidate}", file=sys.stderr)
                continue
            aliases[candidate] = db
    for alias, db in MANUAL_ALIASES.items():
        if alias in dbnames:
            sys.exit(f"manual alias {alias} collides with a real dbname")
        aliases[alias] = db

    for dbname, subdomain in sorted(dbnames.items()):
        print(f"{dbname}\t{subdomain}")
    for alias, db in sorted(aliases.items()):
        print(f"{alias}\t{dbnames[db]}")


if __name__ == "__main__":
    main()
