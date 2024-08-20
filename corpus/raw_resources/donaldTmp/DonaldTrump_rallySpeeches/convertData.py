import csv
import json
from datetime import datetime
import os
import re

arr = os.listdir()

with open('DonaldTrumpspeeches.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    field = ["category", "name", "date", "text"]
    writer.writerow(field)
    for f in arr:
        if (re.search("^.*\.txt$", f)):
            with open(f, "r") as file:
                sizeName = len(f)
                date = f[sizeName - 8:sizeName - 4]
                writer.writerow(["speech", "Donald Trump", date, file.read()])
