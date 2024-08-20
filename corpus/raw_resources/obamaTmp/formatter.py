import csv
import json
from datetime import datetime
import os

def csv_to_json(objInfo):
    csv_file = objInfo["csv_file"]
    data = {"category": objInfo["category"], "content": []}

    json_file = os.path.splitext(csv_file)[0] + ".json"
    with open(csv_file, 'r', newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
    
        for row in reader:
            date = datetime.strptime(row[objInfo["dateCol"]], objInfo["dateFormat"])
            formattedDate = date.strftime("%Y/%m/%d %H:%M")

            content_item = {
                "name": objInfo["name"],
                "date": formattedDate,
                "text": row[objInfo["nameColText"]]
            }
            data['content'].append(content_item)

    with open(json_file, 'w', encoding="utf8") as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)


obj = {
    "csv_file": "Tweets-BarackObama.csv",
    "name": "obama",
    "category": "tweet",
    "dateFormat": "%Y/%m/%d_%H:%M",
    "nameColText": "Tweet-text",
    "dateCol": "Date"
}

csv_to_json(objInfo=obj)
