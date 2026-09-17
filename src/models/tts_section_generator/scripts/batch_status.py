"""
Progress of a running (or finished) batch generation, from its log.

    python3 batch_status.py ~/tts/batch_results.jsonl [total_articles]

Reads the append-only results log the batch script writes. Safe to run
while the batch is running: it only reads, and a partially written last
line is skipped.
"""

import json
import sys
from collections import Counter
from datetime import datetime, timezone

path = sys.argv[1]
total_articles = int(sys.argv[2]) if len(sys.argv) > 2 else None

status = Counter()
manifests, fails, times = 0, [], []
for line in open(path, encoding="utf-8"):
    try:
        r = json.loads(line)
    except ValueError:
        continue  # last line may be mid-write
    s = r.get("status")
    status[s] += 1
    if s == "manifest":
        manifests += 1
    elif s == "fail":
        fails.append(f"{r.get('title')} :: {r.get('section_id')}")
    if r.get("ts"):
        times.append(r["ts"])

print(f"sections: {status['ok']} ok, {status['skip']} skip, {status['fail']} fail")
print(
    f"articles complete (manifest written): {manifests}"
    + (f" of {total_articles}" if total_articles else "")
)

if times:
    t0 = datetime.fromisoformat(min(times))
    t1 = datetime.fromisoformat(max(times))
    elapsed = (t1 - t0).total_seconds()
    done = status["ok"] + status["skip"]
    rate = done / elapsed * 60 if elapsed > 0 else 0
    print(
        f"elapsed {elapsed / 3600:.1f} h; {rate:.1f} sections/min "
        f"(last record {t1.astimezone(timezone.utc):%Y-%m-%d %H:%M} UTC)"
    )
    if total_articles and manifests and rate:
        remaining = (total_articles - manifests) * (done / manifests)
        print(f"projected remaining: {remaining / rate / 60:.1f} h")

if fails:
    print(f"\ndead letters ({len(fails)}):")
    for f in fails[:10]:
        print("  ", f)
    if len(fails) > 10:
        print(f"   ... and {len(fails) - 10} more")
