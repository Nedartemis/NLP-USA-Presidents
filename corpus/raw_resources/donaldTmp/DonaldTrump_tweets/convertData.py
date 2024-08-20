import csv
import json
from datetime import datetime


with open('DonaldTrumpTweets.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["category", "name", "date", "text"]
    writer.writerow(field)
    with open('tweets.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:  
            if row['isRetweet'] == "f":
                writer.writerow(["tweet", "Donald Trump", row["date"], row["text"]])

