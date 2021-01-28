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
    result = [word for word in lowercase.split() if word not in stopwords.words('english')+['able','can','want','would like', 'want to','dont want']]
    result = ' '.join(result)
    return result

#test extract actor
nouns = []
sent ="As an Administrator, I want to be able to reset a User's password for them"
req=(sent)
nlp = spacy.load("en_core_web_sm")
req_doc = nlp(req)   

for tok in req_doc:
    if (tok.dep_ == 'compound' and tok.i == 2): # the idx of the actor is always = 2
        compound = req_doc[tok.i: tok.head.i + 1]
        print("idx :",tok.i)
        actor = compound
        print("actor compound:",actor)
        
    elif(tok.pos_ == 'NOUN' and tok.i == 2):
        actor = tok
        print("actor noun", actor)
        
    
#    token.pos_ == 'NOUN'


#print (req_doc) 
#nounChunks = list(req_doc.noun_chunks)
#print(nounChunks)
#print(nounChunks[0])
#for chunk in nounChunks:
#    print ("chunks all :",chunk)
#    print ("chunks[0] :",chunk[0])


    

#def get_actor(sent):
#    req=(sent)
#    req_doc = textacy.make_spacy_doc(req, lang='en_core_web_sm')
#    #Extract Noun Phrase to explain what nouns are involved
#    actor = ''
#    actors = []
#    for chunk in req_doc.noun_chunks:
#         print ("chunks :",chunk)
#         if(chunk not in actors):
#             actors.append(chunk)
#         actor = str(actors[0]).replace('a ','')
#         actor = actor.replace('an ','')
#         print('actor :', actor)
#             
#             
#    #print(actors[0])
#    return actor


def get_actor(sent):
    actor=''
    req=(sent)
    nlp = spacy.load("en_core_web_sm")
    req_doc = nlp(req)
    for tok in req_doc:
        if (tok.dep_ == 'compound' and tok.i == 2): # the idx of the actor is always = 2
            compound = req_doc[tok.i: tok.head.i + 1]
            print("idx :",tok.i)
            actor = str(compound)
            print("actor compound:",actor)
            
        elif(tok.pos_ == 'NOUN' and tok.i == 2):
            actor = str(tok)
            print("actor noun", actor)
#    req_doc = textacy.make_spacy_doc(req, lang='en_core_web_sm')
    #Extract Noun Phrase to explain what nouns are involved
#    nounChunks = list(req_doc.noun_chunks)
#    print("all nounChunks :",nounChunks)
#    actor = str(nounChunks[0])
#    print("actor:",actor)
#    for chunk in list(req_doc.noun_chunks):
#         print ("req_doc.noun_chunks :",req_doc.noun_chunks)
#         print ("chunks all :",chunk)
##         print ("chunks :",chunk[0])
#         if(chunk not in actors):
#             actors.append(chunk)
#         actor = str(actors[0])
#         print ("actor :", actor)
    return actor

#eliminate the actor
    
def topic_words_extraction(requirement):
    actor_removed_requirement = ''

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
    # # if idf value of word w is too low, it means the word w appears in the majority of preprocessed requirements.
#    for sentence in requirement:
#        clean_sentence = ''
    actor_to_remove = get_actor(requirement.lower())
    print ('actor_to_remove', actor_to_remove)
    actor_removed_requirement = requirement.replace(actor_to_remove,'')
    print ('actor_removed_requirement', actor_removed_requirement)
        
    return actor_removed_requirement

    
if __name__ =="__main__":
    

     
#    df = open('C:/Users/TK257812/Desktop/docs/25-09-19/Visitor.txt','r').readlines()
#    df_preprocessed= open("C:/Users/TK257812/Desktop/docs/25-09-19/preprocessed_visitor.txt",'w')
#    df = open('C:/Users/TK257812/Desktop/docs/25-09-19/User.txt','r').readlines()
#    df_preprocessed= open("C:/Users/TK257812/Desktop/docs/25-09-19/preprocessed_user.txt",'w')
#    df = open('C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/WebCompany.txt','r').readlines()
#    df_preprocessed= open("C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/preprocessed_WebCompany_1.txt",'w')
#    df = open('C:/Users/TK257812/Desktop/docs/25-09-19/admin-original.txt','r').readlines()
#    df_preprocessed= open("C:/Users/TK257812/Desktop/docs/25-09-19/preprocessed_admin.txt",'w')
#    df = open('C:/Users/TK257812/Desktop/docs/15-01-2020/user-stories-examples/PlanningPoker.txt','r').readlines()
#    df_preprocessed= open("C:/Users/TK257812/Desktop/docs/15-01-2020/user-stories-examples/preprocessed_PlanningPoker_1.txt",'w')
#    df = open('C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/CMScompany.txt','r').readlines()
#    df_preprocessed= open("C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/preprocessed_CMScompany_test_4.txt",'w')
    #a= preprocess(df)
#    df = open('C:/Users/TK257812/Desktop/docs/18-05-20/ground truth clusters/web-company.txt','r').readlines()
#    df_preprocessed= open("C:/Users/TK257812/Desktop/docs/18-05-20/ground truth clusters/preprocessed_web-company.txt",'w')
#    df = open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/e-store.txt','r').readlines()
#    df_preprocessed= open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/preprocessed-e-store-2.txt','w')
#    df = open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/inventory.txt','r').readlines()
#    df_preprocessed= open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/preprocessed-inventory-updated.txt','w')
#    df = open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/case-studies-extracting-topic-words/MHC-PMS.txt','r').readlines()
#    df_preprocessed= open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/case-studies-extracting-topic-words/MHC-PMS-preprocessing.txt','w')
    df = open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/wasp.txt','r').readlines()
    df_preprocessed= open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/wasp-preprocessing.txt','w') 
    remove_actor_set = []
    for line in df:
        line = topic_words_extraction (line)
        print ("clean_requirements :",line)
        remove_actor_set.append(line)
        
    print("remove_actor_set : ", remove_actor_set)
        
#        
#        for sentence in clean_requirements:
#            df_preprocessed.write(sentence +"\n") 
#            print(sentence)
   
    
    preprocessed_req = []
    a = lemmatize_sentence("the alarm is set")
    for line in remove_actor_set:
        sentence = preprocessing(line)
        df_preprocessed.write(sentence +"\n") 
        print(sentence)
        
#    print(preprocessed_req)
   
        
    df_preprocessed.close()
    end = time.ctime()
    
    print ("start = ", start)
    print ("end = ", end)
#    df.close()
#    
        
