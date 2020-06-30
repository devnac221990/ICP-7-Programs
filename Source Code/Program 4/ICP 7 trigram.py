import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

file = open("C:/Users/Devna Chaturvedi/PycharmProjects/untitled/inputfinal.txt",encoding='utf-8').read()

token = nltk.word_tokenize(file)

trigrams = ngrams(token,3)


print (Counter(trigrams))
