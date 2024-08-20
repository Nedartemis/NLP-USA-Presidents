import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm


def get_all_files_from_tree(root_tree: str):
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(root_tree)
        for file in files
    ]


def main():
    report = {}
    path_report = "report.json"
    dir_resources = "resources"
    print(f"Checking in '{dir_resources}' tree")

    files = get_all_files_from_tree(dir_resources)

    accepted_exts = [".csv"]

    files = [file for file in files if Path(file).suffix in accepted_exts]

    content_fields = ["category", "name", "date", "text"]

    files_not_good_format: Dict[str, Any] = {}
    for file in tqdm(files):
        print("File : ", file)
        with open(file, "r") as file_reader:
            try:
                infos: pd.DataFrame = pd.read_csv(file)
            except:
                files_not_good_format[file] = ["Cannot be parsed as a csv file"]
                continue
        issues: List[str] = []
        for field in content_fields:
            if field not in infos.columns:
                issues.append("Must contain 'category' field")

        wrong_date_formats: Dict[int, str] = {}
        for index, date in enumerate(infos["date"]):
            try:
                pd.to_datetime(date)
            except KeyboardInterrupt as e:
                raise e
            except:
                wrong_date_formats[index] = date

        if wrong_date_formats:
            issues.append({"At least one date is invalid": wrong_date_formats})

        if issues:
            files_not_good_format[file] = issues

    if files_not_good_format:
        report["files_not_good_format"] = files_not_good_format

    if report:
        print(f"Issue in '{dir_resources}', report saved in '{path_report}'")
        print("Tu veux casser notre dataset en fait ??")
        with open(path_report, "w") as file_writer:
            json.dump(report, file_writer)
    else:
        print("Dataset is all good")


if __name__ == "__main__":
    main()
