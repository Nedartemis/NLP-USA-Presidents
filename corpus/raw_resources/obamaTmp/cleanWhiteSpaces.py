import os

files = os.listdir('barackObamaSpeeches')

for file in files:
    file = 'barackObamaSpeeches/' + file
    # file = 'barackObamaSpeeches/E-Barack-Obama-Speech-Manassas-Virgina-Last-Rally-2008-Election.txt'
    with open(file, 'r+') as readfile:
        res = ""
        for line in readfile.readlines():
            if line.isspace():
                continue
            # print(line.strip())
            res += (line.strip() + " ")
        
    readfile.close()
    with open(file, 'w') as writefile:
        writefile.write(res)
        writefile.close()
