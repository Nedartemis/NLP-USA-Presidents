import csv
import os

arr = os.listdir()


with open("JoeBidenSpeeches.csv", "w", encoding="utf8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["category", "name", "date", "text"])
    for file in arr:
        if file.endswith(".txt"):
            with open(file) as f:
                data = f.read()
            writer.writerow(["speech", "Joe Biden", "2020-09-21 10:00", data.strip()])
