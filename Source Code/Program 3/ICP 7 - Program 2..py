import urllib

import requests
from bs4 import BeautifulSoup

url = ('https://en.wikipedia.org/wiki/Google')
res = requests.get(url)
text = ("downloaded")
html_page = res.content
soup = BeautifulSoup(html_page, 'html.parser')
text = soup.find_all(text=True)

output = ''
blacklist = [
    '[document]',
    'noscript',
    'header',
    'html',
    'meta',
    'head',
    'input',
    'script',
    # there may be more elements you don't want, such as "style", etc.
]

for t in text:
    if t.parent.name not in blacklist:
        output += '{} '.format(t)

print(output)


file = open("inputfinal.txt", 'w+', encoding='utf-8')
source_code = urllib.request.urlopen(url)
plain_text = source_code
soup = BeautifulSoup(plain_text, "html.parser")
file.write(soup.body.text.encode('utf-8', "rb").decode('utf-8'))
file.close()


