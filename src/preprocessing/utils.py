import re

def convertURL(line: str):
    regUrl = r'''https?:\/\/\S+'''
    if re.search(regUrl, line):
        return re.sub(regUrl, "TOKURL", line)
    return line
