import nltk
nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
file=open("C:/Users/Devna Chaturvedi/PycharmProjects/untitled/inputfinal.txt",encoding='utf-8')
file.read()


text_file=nltk.corpus.gutenberg.words("C:/Users/Devna Chaturvedi/PycharmProjects/untitled/inputfinal.txt")
my_lines_list=[]
for line in text_file:
    my_lines_list.append(line)
my_lines_list
print(my_lines_list)