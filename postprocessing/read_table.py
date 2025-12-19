import json

import mlflow
import pandas as pd


def read_json_table(uri: str) -> pd.DataFrame:
    table_path = mlflow.artifacts.download_artifacts(uri)

    with open(table_path) as file:
        json_data = json.load(file)

    df = pd.DataFrame(json_data["data"], columns=json_data["columns"])
    return df
