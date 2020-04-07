# -*- coding: utf-8 -*-

#re regular expression /regex
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize, pos_tag, sent_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import numpy
import time

start = time.ctime()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
lemma = WordNetLemmatizer()
global str
#del str

import cachetools
from cachetools import cached
import spacy
# Set up spaCy
#from spacy.en import English
import textacy
nlp = spacy.load("en_core_web_sm") 



def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    # convert list to string
    list2str = ' '.join(map(str, res))
    return list2str


def delete_punctuation(sentence):
    sentence = re.sub(r"\-", " ", sentence)
    return sentence.translate(str.maketrans('', '', string.punctuation + string.digits))


def preprocessing(sentence):
    # lemmatization
    word_lemmatization = lemmatize_sentence(sentence)
    # delete punctuations
    no_punc = delete_punctuation(word_lemmatization)
    # lower letters,
    lowercase = no_punc.lower()
    # remove stop words and delete Extra Space
    result = [word for word in lowercase.split() if word not in stopwords.words('english')+['able','can','want']]
    result = ' '.join(result)
    return result


def get_actor(sent):
    actor=''
    req=(sent)
    nlp = spacy.load("en_core_web_sm")
    req_doc = nlp(req)
#    req_doc = textacy.make_spacy_doc(req, lang='en_core_web_sm')
    #Extract Noun Phrase to explain what nouns are involved
    actors = []
    for chunk in list(req_doc.noun_chunks)[0]:
         print ("req_doc.noun_chunks :",req_doc.noun_chunks)
         print ("chunks all :",chunk)
#         print ("chunks :",chunk[0])
         if(chunk not in actors):
             actors.append(chunk)
         actor = str(actors[0])
         print ("actor :", actor)
    return actor


def topic_words_extraction(requirement):

    """
    Remove inappropriate words from Lexicon (i.e., topic words).
    1. remove words which don't belong to the the preprocessed requirements;
    2. remove words with very low idf value from lexicon.
    """
    # # idf
    # remove words occuring in more than half the documents
#    vectorizer = TfidfVectorizer(max_df=0.5)
#    vocabulary = vectorizer.fit(requirement).vocabulary_
#    idfMatrix = vectorizer.fit(requirement).idf_
#    print(idfMatrix)
    clean_requirements = []
    # # if idf value of word w is too low, it means the word w appears in the majority of preprocessed requirements.
    for sentence in requirement:
#        clean_sentence = ''
        actor_to_remove = get_actor(sentence)
        sentence = sentence.replace(actor_to_remove,'')
### idf to extract key words
#        for w in sentence.split():
#            #print(w)
#            idfWeight = idfMatrix[vocabulary.get(w)]
#           # print("idfWeight",idfWeight)
#            # print(type(idfWeight))
#            # # if the type of idf value of word w is an array, it means the word w is not in the preprocessed requirements.
#            # # Hence, it should be removed from lexicon (i.e., topic words)
#            if isinstance(idfWeight, numpy.ndarray):
#                continue
#            else:
#                idfWeight = idfMatrix[vocabulary.get(w)]
#                numDoc = len(requirement)
#                print("numDoc",numDoc)
#                ratio_weight = idfWeight / (math.log((numDoc + 1) / 2) + 1)
#                #print("ratio_weight",ratio_weight)
#                if ratio_weight > idf_threshold:
#                    clean_sentence = clean_sentence+' '+w
#                else:
#                    print("excluded word ", w)
                    
#        clean_requirements.append(clean_sentence)
        clean_requirements.append(sentence)
        
    return clean_requirements

    
if __name__ =="__main__":
    

     
#    df = open('C:/Users/TK257812/Desktop/docs/25-09-19/Visitor.txt','r').readlines()
#    df_preprocessed= open("C:/Users/TK257812/Desktop/docs/25-09-19/preprocessed_visitor.txt",'w')
#    df = open('C:/Users/TK257812/Desktop/docs/25-09-19/User.txt','r').readlines()
#    df_preprocessed= open("C:/Users/TK257812/Desktop/docs/25-09-19/preprocessed_user.txt",'w')
    df = open('C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/WebCompany.txt','r').readlines()
    df_preprocessed= open("C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/preprocessed_WebCompany_1.txt",'w')
#    df = open('C:/Users/TK257812/Desktop/docs/25-09-19/admin-original.txt','r').readlines()
#    df_preprocessed= open("C:/Users/TK257812/Desktop/docs/25-09-19/preprocessed_admin.txt",'w')
#    df = open('C:/Users/TK257812/Desktop/docs/15-01-2020/user-stories-examples/PlanningPoker.txt','r').readlines()
#    df_preprocessed= open("C:/Users/TK257812/Desktop/docs/15-01-2020/user-stories-examples/preprocessed_PlanningPoker_1.txt",'w')
#    df = open('C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/CMScompany.txt','r').readlines()
#    df_preprocessed= open("C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/preprocessed_CMScompany_test_4.txt",'w')
    #a= preprocess(df)
    
    preprocessed_req = []
    a = lemmatize_sentence("the alarm is set")
    for line in df:
        sentence = preprocessing(line)
        print(sentence)
        preprocessed_req.append(sentence)
        
    print(preprocessed_req)
    clean_requirements = topic_words_extraction (preprocessed_req)
    for sentence in clean_requirements:
        df_preprocessed.write(sentence +"\n") 
        print(sentence)
        
    df_preprocessed.close()
    end = time.ctime()
    
    print ("start = ", start)
    print ("end = ", end)
#    df.close()
#    
        
