import os

import pandas as pd


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
