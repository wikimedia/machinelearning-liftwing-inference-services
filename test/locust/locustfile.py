import importlib
import os

import pandas as pd

from locust import events
from utils import load_all_results_to_df

model = os.environ.get("MODEL", None)
if model:
    _models = importlib.import_module(f"models.{model}")
    for attr in dir(_models):
        if not attr.startswith("_"):  # Skip internal attributes
            globals()[attr] = getattr(_models, attr)
else:
    from models import *  # noqa


@events.quitting.add_listener
def on_test_end(environment, **_kwargs):
    """
    This function will run after the load test is done, hence we use it to compare the results
    against the stats we have saved in the repository.
    """
    old_results = load_all_results_to_df("results")
    stats_data = []

    for key, entry in environment.stats.entries.items():
        stats_data.append(
            {
                # Add the fields we intend to use
                "Type": entry.method,
                "Name": entry.name,
                "Request Count": entry.num_requests,
                "Failure Count": entry.num_failures,
                "Median Response Time": entry.median_response_time,
                "Average Response Time": entry.avg_response_time,
                "Min Response Time": entry.min_response_time,
                "Max Response Time": entry.max_response_time,
                "Requests/s": entry.total_rps,
                "Failures/s": entry.total_fail_per_sec,
            }
        )
    new_results = pd.DataFrame(stats_data)
    result = pd.merge(
        old_results, new_results, on="Name", how="inner", suffixes=("_old", "_new")
    )
    result = calculate_stats(result)
    cols_to_keep = [
        "Name",
        "Average Response Time Change %",
        "Median Response Time Change %",
        "Requests/s Change %",
    ]
    result = result[cols_to_keep]

    result = strip_name_column(result)
    print_rows_above_threshold(result)


def calculate_stats(result: pd.DataFrame):
    result["Average Response Time Change %"] = calculate_percent_change(
        result["Average Response Time_old"], result["Average Response Time_new"]
    )

    result["Median Response Time Change %"] = calculate_percent_change(
        result["Median Response Time_old"], result["Median Response Time_new"]
    )

    result["Requests/s Change %"] = calculate_percent_change(
        result["Requests/s_old"], result["Requests/s_new"]
    )
    return result


def calculate_percent_change(old_value: float, new_value: float):
    return round((new_value - old_value) * 100 / old_value, 1)


def strip_name_column(result: pd.DataFrame) -> pd.DataFrame:
    pattern = r"/models/(.*?):predict"
    result["Name"] = result["Name"].str.extract(pattern)
    return result


def print_rows_above_threshold(result: pd.DataFrame, threshold_value: int = 10):
    rows_above_threshold = result[
        ~(result.drop("Name", axis=1) < threshold_value).all(axis=1)
    ]
    if rows_above_threshold.empty:
        print("Load test results are within the threshold")
    else:
        raise AssertionError(rows_above_threshold)
