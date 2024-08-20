import csv
import os

arr = os.listdir('barackObamaSpeeches')

with open("barackObamaSpeeches/BarackObamaSpeeches.csv", "w", encoding="utf8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["category", "name", "date", "text"])
    for file in arr:
        if file.endswith(".txt"):
            with open('barackObamaSpeeches/' + file) as f:
                data = f.read()
            writer.writerow(["speech", "Barack Obama", "2020-09-21 10:00", data.strip()])