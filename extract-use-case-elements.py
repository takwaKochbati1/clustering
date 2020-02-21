# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:17:11 2019

@author: TK257812
"""

import re
matches = ""
import collections

s = "As a newly registred User I'm able to open the interactive map from a person's profile page"
a=""
b=""
useCase = ""

def find_start_pattern(sent):
    if "I'm able to" in sent:
        return "I'm able to"
    elif "I am able to" in sent:
        return "I am able to"
    elif "I want to"in sent:
        return "I want to"
    elif "I want"in sent:
        return "I want"
    elif "I need" in sent:
        return "I need"
    elif "I can" in sent:
        return "I can"
    elif "I'm " in sent:
        return "I'm "
    elif "I " in sent:
        return "I "
    else: 
        return -1
    
def find_end_pattern(sent):
    if "so that" in sent:
        return "so that"
    elif "so" in sent:
        return "so"
    else: 
        return ""
        

#startPatterns = [ "I want to"]
#endPatterns = ["so that"]
#secondDelimiter = {"so that","thereby"}
#
#pat = re.compile('a(.*)b')
#result = pat.search(s).group(1)
def get_useCase(sent):
    matches = ""
    useCase = []
    start_pattern = find_start_pattern(sent)
    if(start_pattern != -1):
        match_start=0
        if re.search(start_pattern,  sent):
            a= start_pattern
            match_start=1
            print ('found start match!', start_pattern, match_start)
        else:
            print ('no start match')
            
    else:
        match_start=0
        
            
    end_pattern = find_end_pattern(sent)
    if (end_pattern != ""):
        match_end=0
        if re.search(end_pattern, sent):
            b= end_pattern
            match_end=1
            print ('found end a match!', end_pattern, match_end)
        else:
            print ('no end match')
            
    else:
        match_end=0
        
    
    if(match_start ==1 and match_end ==1):
        matches=re.findall(a+"(.*)"+b,sent)
    elif(match_start ==1 and match_end ==0):
        matches=re.findall(a+"(.*)",sent)
    useCase=matches
    #print(useCase)
    return useCase

#extract verb and interface from use case

import cachetools
from cachetools import cached
import spacy
# Set up spaCy
#from spacy.en import English
import textacy
nlp = spacy.load("en_core_web_sm") 

#def get_verbs(sent):
#    req=(sent)
#    about_talk_text = (get_useCase(sent)[0])
#    #with pos: ADP
#    #nlp("always")[0].pos_
#    pattern = r'(<VERB>?(<ADV>+)*<VERB>+)'
#    #ok pattern = r'(<VERB>?(<ADV>+)*<VERB>+<DET>?<NOUN>+<ADP>?<DET>?<NOUN>+<ADJ>?<NOUN>+)'
#    #pattern = r'(<VERB>?<ADV>*<VERB>+(<DET>?<NOUN>+<ADJ>?<NOUN>+)<PROPN>+(<ADP>?<DET>?)<NOUN>+)'
#    req_doc = textacy.make_spacy_doc(req, lang='en_core_web_sm')
#    about_talk_doc = textacy.make_spacy_doc(about_talk_text, lang='en_core_web_sm')
#    verb_phrases = textacy.extract.pos_regex_matches(about_talk_doc, pattern)
#    # Print all Verb Phrase
#    verbs = []
#    for chunk in verb_phrases:
#         print(chunk.text)
#         verbs.append(chunk.text)
#    #print(verbs)
#    return verbs

def get_nouns(sent):
    about_talk_text = (get_useCase(sent)[0])
    about_talk_doc = textacy.make_spacy_doc(about_talk_text, lang='en_core_web_sm')
    #Extract Noun Phrase to explain what nouns are involved
    chunks=[]
    for chunk in about_talk_doc.noun_chunks:
         print (chunk)
         chunks.append(chunk)
    #print(chunks)
    return chunks
#print("actor:", chunks[0])
    
def get_actor(sent):
    req=(sent)
    req_doc = textacy.make_spacy_doc(req, lang='en_core_web_sm')
    #Extract Noun Phrase to explain what nouns are involved
    actors = []
    for chunk in req_doc.noun_chunks:
         print ("chunks :",chunk)
         if(chunk not in actors):
             actors.append(chunk)
         actor = str(actors[0]).replace('a ','')
         actor = actor.replace('an ','')
             
             
    #print(actors[0])
    return actor


def read_file(file):
    sentences = collections.defaultdict(list)
    req = []
    with open(file) as fin:
        for line in fin:
            if line.strip():
                req.append(line)
                if 'cluster' in line:
                    split = line.split()
                    cluster_number = split[split.index('cluster') + 1]
                if 'cluster' not in line:
                    sentences[cluster_number].append(line)
            
#    fin.close()
    print (dict(sentences))
    return dict(sentences)
    
 

if __name__ == '__main__':
#     cluster = ["As a User I'm able to open the interactive map from a person's profile page so that I can see that particular plot location.", 
#           "As a User I'm able to click a particular plot location from the map so that I can perform a search of people associated with that plot number.",              
#           "As a User I'm able to view an interactive map of the Event Region so that I can view intact mass event locations listed by plot #.",        
#           "As a User I'm able to view an interactive map of the Event Region so that I can find event locations."]
#     cluster_number = 1
#     for s in cluster:
     print("main actor", get_actor("As a User I'm able to open the interactive map from a persons profile page so that I can see that particular plot location.  \n"))
     
     #prob
     #print("main verb", get_verbs("As a User I'm able to open the interactive map from a persons profile page so that I can see that particular plot location.  \n"))
     print("main noun", get_nouns("As a User I'm able to open the interactive map from a persons profile page so that I can see that particular plot location.  \n"))
     print("main use case", get_useCase("As a marketeer     I want to solve url conflicts immediately so I avoid not-friendly URLs and thereby will postively influence the overall SE ranking of the website. "))
#         print("main verbs", get_verbs(s)[0])
#         print("main nouns", get_nouns(s))
#         print("main use case", get_useCase(s)[0])
    
    ## code
     import csv
#     with open('C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/cms-company-extracted-elements.csv', 'w', newline='') as csvfile:
     with open('C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/webCompany-extracted-elements.csv', 'w', newline='') as csvfile:
         fieldnames = ['cluster', 'actor','use_case','interfaces']
         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
         writer.writeheader()
    
#         dictionary = read_file('C:/Users/TK257812/Desktop/docs/25-09-19/clustering-results/cms-company-clusters.txt')
         dictionary = read_file('C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/generated-3-clusters-webCompany.txt')
         print (list(dictionary.keys())[-1])
         cluster_number = int(list(dictionary.keys())[-1])
         print(range(cluster_number))
     
         for cluster in range(cluster_number+1):
             print (cluster)
             print (dictionary[str(cluster)])
             for sentence in dictionary[str(cluster)]:
                 print (sentence)
                 writer.writerow({'cluster': cluster +1, 'actor': get_actor(sentence),'use_case':get_useCase(sentence)[0],
                       'interfaces':get_nouns(sentence)})
     #'verbs':get_verbs(sentence)

     csvfile.close()
             
               
         
            
            

