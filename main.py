# Importing necessary libraries 
import numpy as np
import nltk
import string
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

lemmatizer = nltk.stem.WordNetLemmatizer()


class ChatBot:

    def __init__(self, data_corpus):
        self.greet_inputs = ("hello", "hi", "greetings", "sup","whats up", "hey")
        self.greet_responses = ["hi", "hey", "*nods*", "hi there","I am glad! You are talking to me."]
        self.data_corpus = open(data_corpus, 'r', errors='ignore')
        self.data = self.data_corpus.read().lower()
        self.sent_tokens = self.sent_tokenize(self.data) 
        self.word_tokens = self.word_tokenize(self.data)
        self.remove_puncts = dict((ord(punct), None) for punct in string.punctuation)

    @classmethod
    def sent_tokenize(cls, input_string):
        """
        perform sentence based tokenization.
        """
        sentence_tokens = nltk.sent_tokenize(input_string)
        return sentence_tokens
    
    @classmethod
    def word_tokenize(cls, input_string):
        """
        perform word based tokenization.
        """
        word_tokes = nltk.word_tokenize(input_string)
        return word_tokes

    # def text_preprocessing(self):
        

    def greet(self, sentence):
        
        for word in sentence.split():
            if word.lower() in self.greet_inputs:
                return random.choice(self.greet_responses)
            else:
                pass

    def LemTokens(self, tokens):
        return [lemmatizer.lemmatize(token) for token in tokens]

    def LemNormalize(self, text):
        return self.LemTokens(nltk.word_tokenize(text.lower().translate(self.remove_puncts)))

    def response(self, user_response):
        robo1_response = ""
        TfidfVec = TfidfVectorizer(tokenizer=self.LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(self.sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idf = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if (req_tfidf==0):
            robo1_response = robo1_response+"I am sorry! I don't understand you."
            return robo1_response
        else:
            robo1_response = robo1_response+self.sent_tokens[idf]
            return robo1_response

    def start_chat(self):
        flag = True
        print("BOT: My name is stark. Lets have a conversation! Also, if you want to exit anytime, just type Bye!")

        while(flag==True):
            user_response = input()
            user_response = user_response.lower()
            if (user_response!='bye'):
                if (user_response=='thanks' or user_response=='thank you'):
                    flag=False
                    print("BOT: You are Welcome...")
                else:
                    if self.greet(user_response)!=None:
                        print("BOT: "+self.greet(user_response))
                    else:
                        self.sent_tokens.append(user_response)
                        word_tokens = self.word_tokens+nltk.word_tokenize(user_response)
                        final_words = list(set(word_tokens))
                        print("BOT: ",end="")
                        print(self.response(user_response))
                        self.sent_tokens.remove(user_response)
            else:
                flag=False
                print("BOT: Goodbye! Take cate <3 ")

chatbot = ChatBot('/home/data-engineer/chatbot.txt')
# print(chatbot.sent_tokenize(chatbot.data))
# print(chatbot.word_tokenize(chatbot.data))
chatbot.start_chat()



