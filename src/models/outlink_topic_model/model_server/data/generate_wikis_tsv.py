#!/usr/bin/env python3
"""Regenerate wikis.tsv (wiki_id -> wikipedia subdomain) from the sitematrix API.

The second column is the *served subdomain* of {sub}.wikipedia.org, used by the
outlink topic model as the MW API Host header. It is NOT the ISO language code:
the two differ for e.g. alswiki (als, not gsw) and be_x_oldwiki (be-tarask).

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


def main() -> None:
    req = urllib.request.Request(SITEMATRIX_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as resp:
        sitematrix = json.load(resp)["sitematrix"]

    rows = []
    for key, group in sitematrix.items():
        if key == "count":
            continue
        sites = group if key == "specials" else group.get("site", [])
        for site in sites:
            if site.get("code") != "wiki":  # Wikipedia family only
                continue
            host = site["url"].removeprefix("https://")
            if not host.endswith(".wikipedia.org"):
                print(f"skipping {site['dbname']}: {site['url']}", file=sys.stderr)
                continue
            rows.append((site["dbname"], host.removesuffix(".wikipedia.org")))

    for dbname, subdomain in sorted(rows):
        print(f"{dbname}\t{subdomain}")


if __name__ == "__main__":
    main()
