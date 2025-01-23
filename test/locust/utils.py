import os
import sys
import argparse
import requests
import pandas as pd
from distutils.util import strtobool


def load_all_results_to_df(root_dir):
    all_dfs = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith("_stats.csv"):
                file_path = os.path.join(subdir, file)
                try:
                    df = pd.read_csv(file_path)
                except pd.errors.EmptyDataError:
                    continue
                df = df[df["Name"] != "Aggregated"]  # filter out aggregated rows
                all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def get_recent_revision_info(limit, language, save_to_file=False):
    """
    Fetches the revision IDs and titles of the most recent changes to Wikipedia.

    :param limit: The number of recent changes to fetch.
    :param language: The language of the Wikipedia article.
    :return: A list of tuples containing revision IDs and article titles.
    """
    url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "recentchanges",
        "rcprop": "ids|title",
        "rclimit": limit,
        "rcnamespace": 0,  # 0 is the main namespace
        "format": "json",
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Return a list of tuples (language, revision_id, title),
    # filtering out revisions with ID 0
    revision_info = [
        (language, change["revid"], change["title"])
        for change in data["query"]["recentchanges"]
        if change["revid"] != 0  # filter out revisions with ID 0
    ]
    if strtobool(save_to_file):
        df = pd.DataFrame(revision_info)
        df.columns = ["lang", "rev_id", "title"]
        df.to_csv(f"data/recentchanges_{language}.tsv", sep="\t", index=False)
    return revision_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str)
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()

    if hasattr(sys.modules[__name__], args.function):
        func = getattr(sys.modules[__name__], args.function)
        result = func(*args.args)
    else:
        print(f"Function {args.function} not found in utils.py")
