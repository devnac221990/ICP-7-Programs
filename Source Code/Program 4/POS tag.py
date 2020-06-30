import nltk

from nltk.tokenize import word_tokenize
file_content = open("C:/Users/Devna Chaturvedi/PycharmProjects/untitled/inputfinal.txt",encoding='utf-8').read()

nltk.pos_tag(file_content)
print(nltk.pos_tag(file_content))
