import re
import csv
from datetime import datetime

def parseDebate(tup: str):
    filein, datestr = tup
    date = datetime.strptime(datestr, "%Y-%m-%d %H:%M")

    with open(filein) as file:
        l = re.split(r'(LEHRER|OBAMA|ROMNEY|CROWLEY|SCHIEFFER): ', file.read())
        res = []
        i = 0
        while i < len(l):
            if l[i] == "OBAMA":
                striped = re.sub(r'\([A-Z]+\)', '', l[i+1])
                striped = re.sub(r'\n', '', striped)
                res.append(striped)
                i += 1
            i += 1
        
    outputName = filein.split('.')[0] + '.csv'
    with open(outputName, 'w', encoding="utf8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['category', 'name', 'date', 'text'])
        for elm in res:
            writer.writerow(['debate', 'Barack Obama', date, elm])


folderDebate = "barackObamaDebate/"

debates = [(folderDebate + "debate-romney-obama1.txt", "2012-10-3 09:00"),
            (folderDebate + "debate-romney-obama2.txt", "2012-10-16 09:00"),
            (folderDebate + "debate-romney-obama3.txt", "2012-10-22 09:00")]
            
for debate in debates:
    parseDebate(debate)


files = ["debate-romney-obama1.csv",
         "debate-romney-obama2.csv",
         "debate-romney-obama3.csv",]

def mergeCSV(files):
    with open("barackObamaDebate/barackDebates.csv", 'w') as writefile:
        writer = csv.writer(writefile)
        writer.writerow(['category', 'name', 'date', 'text'])
        for file in files:
            with open("barackObamaDebate/" + file, 'r') as readfile:
                lines = readfile.readlines()
            readfile.close()
            writefile.write(''.join(lines[1:]))
    writefile.close()


mergeCSV(files)