# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:44:20 2019

@author: takwa
"""
from scipy.cluster.hierarchy import ward, fcluster, leaders,linkage,dendrogram 
from scipy.spatial.distance import pdist
import collections
import numpy as np
import gensim.models as gm
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import matutils
#re regular expression /regex
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from gensim.summarization import keywords
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc 
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from dunn_index import dunn
import time
#import sys
#sys.path.append('C:/UsersTK257812/requirements clustering/dunn_index.py')
#from jqmcvi import base
start = time.ctime() 
nltk.download('stopwords')
nltk.download('wordnet')

lemma = WordNetLemmatizer()
stopword_set = set(stopwords.words('english')+['a','of','at','s','for'])


 
#df = open('C:/Users/TK257812/Desktop/docs/25-09-19/preprocessed_admin.txt','r').readlines()
#df_orig = open('C:/Users/TK257812/Desktop/docs/25-09-19/admin-original.txt','r').readlines()
#df = open('C:/Users/TK257812/Desktop/docs/25-09-19/preprocessed_visitor.txt','r').readlines()
#df_orig = open('C:/Users/TK257812/Desktop/docs/25-09-19/Visitor.txt','r').readlines()
#df = open('C:/Users/takwa/Desktop/files/25-09-19/user-keyWords.txt','r').readlines()
#df_orig = open('C:/Users/TK257812/Desktop/docs/25-09-19/User.txt','r').readlines()
#df = open('C:/Users/TK257812/Desktop/docs/25-09-19/preprocessed_user.txt','r').readlines()

#df = open("C:/Users/TK257812/Desktop/docs/15-01-2020/user-stories-examples/preprocessed_PlanningPoker_1.txt",'r').readlines()
#df_orig = open('C:/Users/TK257812/Desktop/docs/15-01-2020/user-stories-examples/PlanningPoker.txt','r').readlines()

#df_orig = open('C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/WebCompany.txt','r').readlines()
#df= open("C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/preprocessed_WebCompany_1.txt",'r').readlines()
#df_orig = open('C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/CMScompany.txt','r').readlines()
#df= open("C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/preprocessed_CMScompany.txt",'r').readlines()
#df_orig = open('C:/Users/TK257812/Desktop/docs/18-05-20/ground truth clusters/web-company.txt','r').readlines()
#df= open("C:/Users/TK257812/Desktop/docs/18-05-20/ground truth clusters/preprocessed_web-company.txt",'r').readlines()
#df_orig = open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/e-store.txt','r').readlines()
##df= open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/preprocessed-e-store.txt','r').readlines()
#df= open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/preprocessed-e-store-2.txt','r').readlines()
#df_orig = open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/case-studies-extracting-topic-words/MHC-PMS.txt','r').readlines()
#df= open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/preprocessed-inventory-updated.txt','r').readlines()
#df= open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/case-studies-extracting-topic-words/MHC-PMS-preprocessing.txt','r').readlines()
df_orig = open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/wasp.txt','r').readlines()
df= open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/wasp-preprocessing.txt','r').readlines()

dfLen = len(df)

text = []
for line in df:
    print (line)
    text.append(line) 
print(text)

#tfidf
tfidf_vectorizer = TfidfVectorizer()
# A mapping of terms to feature indices. => dictionnary
vocabulary = tfidf_vectorizer.fit(text).vocabulary_
tfidf = tfidf_vectorizer.fit_transform(text) 
#idf 
idfMatrix = tfidf_vectorizer.fit(text).idf_
#print("fit-transform: ", tfidf) 

#print("idfMatrix : ",idfMatrix) 
#print("specific_word_idf_value: ", idfMatrix[vocabulary.get("set")])

#Clustering k-means
#kmeans = KMeans(n_clusters=3).fit(tfidf)
#print (kmeans.labels_)

#######word2vec similarity computation

model = gm.KeyedVectors.load('C:/Users/TK257812/.spyder-py3/GoogleNews-simplified',mmap='r')  # mmap the large matrix as read-only
model.syn0norm = model.syn0  # no need to call init_sims
## cos similarity ##
dim = model.vector_size  # dimensionality of word embedding
# Out of vocabulary
UNKNOW_WORDS = {}

def cos_similarity(w1, w2, wordEmb, dim):
    '''
    :param w1: word in requirement
    :param w2: word in requirement
    :param wordEmb: The pre-trained word2vec model
    :param dim: The dimensionality of the pre-trained word2vec model.
    :return:
    '''
    if w1 in wordEmb.vocab:
        v1 = wordEmb[w1]
    elif w1 in UNKNOW_WORDS:
        v1 = UNKNOW_WORDS.get(w1)
        print(" unknown word:", w1)
    else:
        embeddings_random = np.random.uniform(-0.1, 0.1, dim)
        UNKNOW_WORDS.update({w1: embeddings_random})
        print(" unknown word:", w1)
        v1 = embeddings_random

    if w2 in wordEmb.vocab:
        v2 = wordEmb[w2]
    elif w2 in UNKNOW_WORDS:
        v2 = UNKNOW_WORDS.get(w2)
        print(" unknown word:",w2)
    else:
        embeddings_random = np.random.uniform(-0.1, 0.1, dim)
        UNKNOW_WORDS.update({w2: embeddings_random})
        print(" unknown word:", w2)
        v2 = embeddings_random

    return np.dot(matutils.unitvec(v1), matutils.unitvec(v2))

#sent1 = "edit exist event update content"
#sent2 = "add new event show list event"
#sent3 = "logout so that other Users of my device don't have access to my private account"
#sent4 = "logged out after 10 minutes or more of inactivity account stays secure"
#sent5 = "view interactive map Event Region find event locations"

#print("cos_similarity : ",cos_similarity(process(sent1)[0], process(sent2)[0],model,dim))
#print("n_similarity:",model.n_similarity(process(sent1).split(), process(sent2).split()))

####  test similarity computation ####
#for w1 in sent1.split():
#    sim1 = []
#    for w2 in sent2.split():
#        sim1.append(cos_similarity(w1, w2, model, dim))
#        print(w1,w2,cos_similarity(w1, w2, model, dim))
#    print("sim1 : ", sim1)
        #idf = idfMatrix[vocabulary.get(w1)]
    
def inner_similarity_mihalcea(sent1, sent2):
    sumSim = 0
    sumIdf = 0
    for w1 in sent1.split():
#        print("sent1 ", sent1)
        sim = []
        for w2 in sent2.split():
            sim.append(cos_similarity(w1, w2, model, dim))
            idf = idfMatrix[vocabulary.get(w1)]
#            print("words idf : ", w1, w2, idf)
#            print("sent2 ", sent2)
        maxSim = max(sim)
#        print("maxSim", maxSim)
        innerSim = idf * maxSim
        #print("idf* maxSim = innerSim", innerSim)
        sumSim += innerSim
        sumIdf += idf
    #print("sumIdf", sumIdf)
    result = sumSim / sumIdf
    return result

def similarity_mihalcea (reqDoc):
    simReq = []
    for reqL in df:
        row = []
        for reqC in df:
            row.append(0.5 * (inner_similarity_mihalcea(reqL, reqC)+ inner_similarity_mihalcea(reqC, reqL)))
            simReq.append(row)
    return simReq

#print("inner_similarity_mihalcea 1 : ", inner_similarity_mihalcea(process(sent5), process(sent1)))
#print("inner_similarity_mihalcea 2: ", inner_similarity_mihalcea(process(sent1), process(sent5)))

final_matrix = similarity_mihalcea (df)
#print("final_matrix : ", final_matrix)


##SILHOUETTE SCORE

#best_clusters = 0                       # best cluster number which you will get
#previous_silh_avg = 0.0
#for n_cluster in range(2,dfLen):
#    clusterer = KMeans(n_clusters=n_cluster)
#    cluster_labels = clusterer.fit_predict(final_matrix)
#    silhouette_avg = silhouette_score(final_matrix, cluster_labels, metric='euclidean')
#    if silhouette_avg > previous_silh_avg:
#        previous_silh_avg = silhouette_avg
#        best_clusters = n_cluster
#print ("best_clusters : ",best_clusters)
#n_clusters=best_clusters

#Clustering k-means
def clustering(n_clusters, final_matrix, file):
    
    kmeans = KMeans(n_clusters,init= 'random', n_init=10,max_iter=300,tol=1e-04, random_state=0).fit(final_matrix)
    print(kmeans.labels_)
    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)
    
    
    print(dict(clusters))
    new_clusters = collections.defaultdict(list)
    
    for cluster in range(n_clusters):
        for j in dict(clusters)[cluster]:
            if j % (len(file)) == 0:
                new_clusters[cluster].append(int(j / len(file)))
    print("new clusters : ", dict(new_clusters))
    
    ## get sentences from the txt file
    return dict(new_clusters)

def clustering_Hac(n_clusters, final_matrix, file):
    
    #Plot the dendogram
    Z = linkage(final_matrix, method='ward', metric='euclidean')
    #BUILD DENDROGRAM, p represents the number of the axis
    dendrogram(Z, truncate_mode= "lastp", p =12, leaf_rotation=45,leaf_font_size=15, show_contracted=True) # visualize the clustering result
    #plt.figure(figsize=(10, 7)) 
    plt.xlabel("Cluster Size")
    plt.ylabel("Distance")
    #cut point
    #plt.axhline(y=6, color='r', linestyle='--')
    plt.show()
    
    #HAC clustering
    HClustering = AgglomerativeClustering(n_clusters , affinity="euclidean",linkage="ward").fit(final_matrix)
    HClustering.fit_predict(final_matrix)
    
        #visualize the clusters
    print("HClustering.labels_lenght :", len(HClustering.labels_))
    print("HClustering.labels :", HClustering.labels_)
    clusters = collections.defaultdict(list)
    
    for i, label in enumerate(HClustering.labels_):
        clusters[label].append(i)
    
    print(dict(clusters))
    new_clusters = collections.defaultdict(list)
    print("range(n_clusters) :", range(n_clusters))
    for cluster in range(n_clusters):
        print("dict(clusters)[cluster]:", dict(clusters)[cluster])
        for j in dict(clusters)[cluster]:
#            print("j = ",j)
#            each line in the matrix has the lenght of the file
            if j % (len(file)) == 0:
#                print("j", j)
                new_clusters[cluster].append(int(j / len(file)))
#            print("new clusters_j : ", dict(new_clusters))
    print("new clusters : ", dict(new_clusters))
    
    ## get sentences from the txt file
    return dict(new_clusters)


#cluster labels
def key_words (cluster):
    a = keywords(cluster, words = 5,scores = True, lemmatize = True)
    print("key words summarization gensim",a)
    #return tfidf_.get_feature_names()
    return a
    
    
if __name__ =="__main__":
    
#    df_generated_clusters= open("C:/Users/TK257812/Desktop/docs/25-09-19/user_stories/CMScompany_generated_clusters_1.txt",'w')
#    df_generated_clusters= open("C:/Users/TK257812/Desktop/docs/18-05-20/ground truth clusters/clustered_web-company.txt",'w')
#    df_generated_clusters= open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/clustered-e-store-2.txt','w')
#    df_generated_clusters= open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/case-studies-extracting-topic-words/clustered-MHC-PMS.txt','w')
    df_generated_clusters= open('C:/Users/TK257812/Desktop/docs/18-05-20/clustering-baseline/clustered-wasp','w')
    aggro_clusters_i = []
    k_dunn_calinski_i = []
    silhouette_scores_i = [] 
# Appending the silhouette scores of the different models to the list
    for i in range (2,10):
        ac = AgglomerativeClustering(i , affinity="euclidean",linkage="ward").fit(final_matrix)
        silhouette_scores_i.append( 
           silhouette_score(final_matrix, ac.fit_predict(final_matrix))) 
        aggro_clusters_i.append(ac.labels_)
        k_dunn_calinski_i.append(i)
    
#    ac8 = AgglomerativeClustering(13 , affinity="euclidean",linkage="ward").fit(final_matrix)
#    ac7 = AgglomerativeClustering(12 , affinity="euclidean",linkage="ward").fit(final_matrix)
#    ac6 = AgglomerativeClustering(11 , affinity="euclidean",linkage="ward").fit(final_matrix)
#    ac5 = AgglomerativeClustering(10 , affinity="euclidean",linkage="ward").fit(final_matrix)
#    ac4 = AgglomerativeClustering(9 , affinity="euclidean",linkage="ward").fit(final_matrix)
#    ac3 = AgglomerativeClustering(8 , affinity="euclidean",linkage="ward").fit(final_matrix)
#    
#    aggro_clusters = [ac3.labels_,ac4.labels_,ac5.labels_,ac6.labels_,ac7.labels_,ac8.labels_]
##    aggro_clusters = [ac2.labels_,ac3.labels_,ac4.labels_,ac5.labels_,ac6.labels_]
##    k_dunn_calinski = [2, 3, 4, 5, 6]
##    k_dunn_calinski = [16, 17, 18, 19, 20, 21]
#    k_dunn_calinski = [8, 9, 10, 11, 12,13]
    
#    ac2= KMeans(2,init= 'random', n_init=10,max_iter=300,tol=1e-04, random_state=0)
#    ac3= KMeans(3,init= 'random', n_init=10,max_iter=300,tol=1e-04, random_state=0)
#    ac4= KMeans(4,init= 'random', n_init=10,max_iter=300,tol=1e-04, random_state=0)
#    ac5= KMeans(5,init= 'random', n_init=10,max_iter=300,tol=1e-04, random_state=0)
#    ac6= KMeans(6,init= 'random', n_init=10,max_iter=300,tol=1e-04, random_state=0)
#    ac7= KMeans(7,init= 'random', n_init=10,max_iter=300,tol=1e-04, random_state=0)
#    ac8= KMeans(8,init= 'random', n_init=10,max_iter=300,tol=1e-04, random_state=0)
#    ac12= KMeans(12,init= 'random', n_init=10,max_iter=300,tol=1e-04, random_state=0)
   
    ##### silhouette score
    
#    silhouette_scores = [] 
#    silhouette_scores.append( 
#            silhouette_score(final_matrix, ac2.fit_predict(final_matrix))) 
#    silhouette_scores.append( 
#            silhouette_score(final_matrix, ac3.fit_predict(final_matrix))) 
#    silhouette_scores.append( 
#            silhouette_score(final_matrix, ac4.fit_predict(final_matrix))) 
#    silhouette_scores.append( 
#            silhouette_score(final_matrix, ac5.fit_predict(final_matrix))) 
#    silhouette_scores.append( 
#            silhouette_score(final_matrix, ac6.fit_predict(final_matrix))) 
#    silhouette_scores.append( 
#            silhouette_score(final_matrix, ac7.fit_predict(final_matrix))) 
#    silhouette_scores.append( 
#            silhouette_score(final_matrix, ac8.fit_predict(final_matrix))) 
#    silhouette_scores.append( 
#            silhouette_score(final_matrix, ac18.fit_predict(final_matrix))) 
##    
#    print("silhouette_scores", silhouette_scores)
#     #get the maximum silhouette score
    maxElement = np.amax(silhouette_scores_i)
    optimal_cluster_number_silhouette = np.argmax(silhouette_scores_i)+2
# 
#    print('Max element from Numpy Array : ', maxElement)
#    print('optimal_cluster_number : ', optimal_cluster_number)
#    
    #####################
    max_dunn_index = 0
    dunn_scores = []
#    calinski_harabaz_scores = []
    for i in aggro_clusters_i:
        score1 = dunn(i, metrics.pairwise.euclidean_distances(final_matrix))
#        score2 = metrics.calinski_harabaz_score(final_matrix, i)
        dunn_scores.append(score1)
        #calinski_harabaz_scores.append(score2)
#        print ("score1",score1)
        
    print("dunn_score: ",dunn_scores)
    #print("calinski_harabaz_scores: ",calinski_harabaz_scores)
    
    maxElement_dunn = np.amax(dunn_scores)
#    optimal_numCluster_dunn = np.argmax(dunn_scores)+2
#    optimal_numCluster_dunn = np.argmax(dunn_scores)+16
    optimal_numCluster_dunn = np.argmax(dunn_scores)+2
    
#HAC  
    #based on the dendrogram we have 5 clusters 
#    new_clusters = clustering_Hac(optimal_numCluster_dunn, final_matrix, df_orig)
    new_clusters = clustering_Hac(14, final_matrix, df_orig)
#    for cluster in range(optimal_numCluster_dunn):
    for cluster in range(14):
        tab_cluster = ""
        print("cluster", cluster,"\n")
        df_generated_clusters.write("cluster "+str(cluster + 1)+"\n\n") 
        for i, sentenceIndex in enumerate(new_clusters[cluster]):
            print("sentence ", i,":", df_orig[sentenceIndex],"\n")
            df_generated_clusters.write(df_orig[sentenceIndex]+"\n") 
            tab_cluster = tab_cluster + str(df[sentenceIndex])
        #key_words (tab_cluster)
        print("sentenceIndex:", sentenceIndex)
        key_words(tab_cluster)
    df_generated_clusters.close()
        
# Plotting a bar graph for dunn scores 
    print("dunn scores",dunn_scores)
    plt.bar(k_dunn_calinski_i, dunn_scores) 
    plt.xlabel('Number of clusters') 
    plt.ylabel('dunn(i)') 
    plt.show()  
    end = time.ctime()
    print ("start = ", start)
    print ("end = ", end)
    
# Plotting a bar graph for calinski_harabaz_scores 
#    print("calinski harabaz scores",calinski_harabaz_scores)
#    plt.bar(k_dunn_calinski, calinski_harabaz_scores) 
#    plt.xlabel('Number of clusters') 
#    plt.ylabel('calinski harabaz scores(i)') 
#    plt.show()  
        


