import csv
from datetime import datetime

with open('barackObamaTweet/Tweets-BarackObama.csv', newline='') as csvfile:
    with open("barackObamaTweet/tweetsBarackObama.csv", "w", encoding="utf8") as outfile:
        reader = csv.DictReader(csvfile)
        writer = csv.writer(outfile)
        writer.writerow(["category", "name", "date", "text"])
        for row in reader:
            date = datetime.strptime(row["Date"], "%Y/%m/%d_%H:%M") 
            formattedDate = date.strftime("%Y/%m/%d %H:%M")
            writer.writerow(['tweet', 'Barack Obama', formattedDate, row["Tweet-text"]])


