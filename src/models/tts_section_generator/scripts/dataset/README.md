# Dataset generation script for TTS v1 experiment

Check https://phabricator.wikimedia.org/T432128 to learn more about the context.

Per topic: 50 seed articles (Featured Articles + a pageview-percentile sample,
filtered by langlinks, edit volatility, keywords and article quality, ranked by
topic score) plus 150 MoreLike recommendations, 200 articles total. Topics
short of 50 seeds are topped up with extra recommendations.

# How to run

The script calls Lift Wing on the internal endpoint, so it has to run on an
internal host (e.g. a stat box). There the public MW API needs the webproxy;
`no_proxy` keeps Lift Wing traffic direct:

```bash
https_proxy=http://webproxy.eqiad.wmnet:8080 no_proxy=wmnet \
    python3 make_dataset.py [topic ...]    # default: all 39 topics
```

Output is appended to `dataset.csv` next to the script. Topics already present
in the file are skipped, so an interrupted run can simply be restarted.

`SAMPLE_SIZE` (default 10000) controls the random sample size per topic.
