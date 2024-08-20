import requests
from bs4 import BeautifulSoup

response = requests.get('http://obamaspeeches.com/E-Barack-Obama-Speech-Manassas-Virgina-Last-Rally-2008-Election.htm')
soup = BeautifulSoup(response.text, 'html.parser')
listhref = soup.table.table

actual_web_links = [web_link['href'] for web_link in listhref.td.select('a')]
print(actual_web_links)

link = "http://obamaspeeches.com"
for links in actual_web_links:
    slash = ""
    if links[0] != "/":
        slash = "/"
    response = requests.get(link + slash +links)
    soup = BeautifulSoup(response.text, 'html.parser')
    list = soup.table.tr.find_all("table")[1].tr.td.find_all('p')[1:]
    if list == []:
        list = soup.table.tr.find_all("table")[1].find_all('td')[2].find_all('p')

    text = ""
    for elm in list:
        text += (elm.get_text() + " ")
    
    filename = (links.split('.')[0])[1:]
    with open("barackObamaSpeeches/" + filename + ".txt", "w") as text_file:
        text_file.write(text)




# response = requests.get('http://obamaspeeches.com/E-Barack-Obama-Speech-Manassas-Virgina-Last-Rally-2008-Election.htm')
# soup = BeautifulSoup(response.text, 'html.parser')
# listhref = soup.table.table

# print(listhref.p.find_all('font')[3].get_text())