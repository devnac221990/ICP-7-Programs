import nltk
file_content = open("C:/Users/Devna Chaturvedi/PycharmProjects/untitled/inputfinal.txt",encoding='utf-8').read()
tokens = nltk.word_tokenize(file_content)
print (tokens)
