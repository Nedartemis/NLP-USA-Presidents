import csv
import re

with open('RawTweetsJoeBiden.csv', newline='') as csvfile:
    with open("JoeBidenTweets.csv", "w", encoding="utf8") as outfile:
        reader = csv.DictReader(csvfile)
        writer = csv.writer(outfile)
        writer.writerow(["category", "name", "date", "text"])
        for row in reader:
            found = re.search("RT @.*", row["tweet"])
            if not found:
                writer.writerow(['tweet', 'Joe Biden', row["timestamp"], row["tweet"]])
