
import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
file_content = open("C:/Users/Devna Chaturvedi/PycharmProjects/untitled/inputfinal.txt",encoding='utf-8').read()

print(ne_chunk(pos_tag(wordpunct_tokenize(file_content))))

