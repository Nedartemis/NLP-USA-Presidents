import csv
import json
from datetime import datetime
import os
import re

os.remove("DonaldTrumpDebates.csv")

arr = os.listdir()

with open('DonaldTrumpDebates.csv', 'w', newline='') as csvFile:
    for f in arr:
        if (re.search("^.*\.csv$", f)):
            with open(f, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                writer = csv.writer(csvFile)
                field = ["category", "name", "date", "text"]
                writer.writerow(field)
                for row in reader:  
                    if (row["speaker"] == "Vice President Joe Biden"):
                        writer.writerow(["debate", "Joe Biden", "2020 " + row['minute'], row['text']])
                    elif (re.search("^.*Trump.*$",row["speaker"])):
                        writer.writerow(["debate", "Donald Trump", "2020 " + row['minute'], row['text']])
                    else:
                        writer.writerow(["debate", row['speaker'], "2020 " + row['minute'], row['text']])
