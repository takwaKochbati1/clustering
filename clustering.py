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
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc 
from sklearn.cluster import AgglomerativeClustering
nltk.download('stopwords')
nltk.download('wordnet')

lemma = WordNetLemmatizer()
stopword_set = set(stopwords.words('english')+['a','of','at','s','for'])


 
#df = open('C:/Users/takwa/Desktop/files/25-09-19/admin-key-words.txt','r').readlines()
#df_orig = open('C:/Users/takwa/Desktop/files/25-09-19/admin-original.txt','r').readlines()
df = open('C:/Users/TK257812/Desktop/docs/25-09-19/Visitor-key-visitor.txt','r').readlines()
df_orig = open('C:/Users/TK257812/Desktop/docs/25-09-19/Visitor.txt','r').readlines()
#df = open('C:/Users/takwa/Desktop/files/25-09-19/user-keyWords.txt','r').readlines()
#df_orig = open('C:/Users/takwa/Desktop/files/25-09-19/User.txt','r').readlines()



#df = open('C:/Users/takwa/Desktop/files/25-09-19/test_docs_tjc.txt','r').readlines()
dfLen = len(df)

## cleaning
def process(string):
    string=' '+string+' '
    string=' '.join([word if word not in stopword_set else '' for word in string.split()])
    
#search and replace
    string=re.sub('\@\w*',' ',string)
    string=re.sub('\.',' ',string)
    string=re.sub("[,#'-\(\):$;\?%]",' ',string)
    string=re.sub("\d",' ',string)
    string=string.lower()
    string=re.sub("nyse",' ',string)
    string=re.sub("inc",' ',string)
    string=re.sub(r'[^\x00-\x7F]+',' ', string)
    string=re.sub(' for ',' ', string)
    string=re.sub(' s ',' ', string)
    string=re.sub(' the ',' ', string)
    string=re.sub(' a ',' ', string)
    string=re.sub(' with ',' ', string)
    string=re.sub(' is ',' ', string)
    string=re.sub(' at ',' ', string)
    string=re.sub(' to ',' ', string)
    string=re.sub(' by ',' ', string)
    string=re.sub(' when ',' ', string)
    string=re.sub(' of ',' ', string)
    string=re.sub(' are ',' ', string)
    string=re.sub(' if ',' ', string)
    string=re.sub(' on ',' ', string)
    string=re.sub(' can ',' ', string)
    string=re.sub(' must ',' ', string)
    string=re.sub(' system ',' ', string)
    string=" ".join(lemma.lemmatize(word) for word in string.split())
    string=re.sub('( [\w]{1,2} )',' ', string)
    string=re.sub("\s+",' ',string)
    return string

#process text 
text = []
for line in df:
    print (line)
    newLine = process(line)
    text.append(newLine) 
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
    else:
        embeddings_random = np.random.uniform(-0.1, 0.1, dim)
        UNKNOW_WORDS.update({w1: embeddings_random})
        v1 = embeddings_random

    if w2 in wordEmb.vocab:
        v2 = wordEmb[w2]
    elif w2 in UNKNOW_WORDS:
        v2 = UNKNOW_WORDS.get(w2)
    else:
        embeddings_random = np.random.uniform(-0.1, 0.1, dim)
        UNKNOW_WORDS.update({w2: embeddings_random})
        v2 = embeddings_random

    return np.dot(matutils.unitvec(v1), matutils.unitvec(v2))

sent1 = "set a new password login"
sent2 = "request a  password reset login forgot password"
sent3 = "logout so that other Users of my device don't have access to my private account"
sent4 = "logged out after 10 minutes or more of inactivity account stays secure"
sent5 = "view interactive map Event Region find event locations"

#print("cos_similarity : ",cos_similarity(process(sent1)[0], process(sent2)[0],model,dim))
#print("n_similarity:",model.n_similarity(process(sent1).split(), process(sent2).split()))

def inner_similarity_mihalcea(sent1, sent2):
    sumSim = 0
    sumIdf = 0
    for w1 in sent1.split():
        sim = []
        for w2 in sent2.split():
            sim.append(cos_similarity(w1, w2, model, dim))
            idf = idfMatrix[vocabulary.get(w1)]
        #print("words idf : ", w1, idf)
        maxSim = max(sim)
        #print("maxSim", maxSim)
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
            row.append(0.5 * (inner_similarity_mihalcea(process(reqL), process(reqC))+ inner_similarity_mihalcea(process(reqC), process(reqL))))
            simReq.append(row)
    return simReq

#print("inner_similarity_mihalcea 1 : ", inner_similarity_mihalcea(process(sent5), process(sent1)))
#print("inner_similarity_mihalcea 2: ", inner_similarity_mihalcea(process(sent1), process(sent5)))

final_matrix = similarity_mihalcea (df)
#print("final_matrix : ", final_matrix)


##SILHOUETTE SCORE

best_clusters = 0                       # best cluster number which you will get
previous_silh_avg = 0.0
for n_cluster in range(2,dfLen):
    clusterer = KMeans(n_clusters=n_cluster)
    cluster_labels = clusterer.fit_predict(final_matrix)
    silhouette_avg = silhouette_score(final_matrix, cluster_labels, metric='euclidean')
    if silhouette_avg > previous_silh_avg:
        previous_silh_avg = silhouette_avg
        best_clusters = n_cluster
print ("best_clusters : ",best_clusters)
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
    dendrogram(Z, truncate_mode= "lastp", p =12, leaf_rotation=45,leaf_font_size=15, show_contracted=True) # visualize the clustering result
    #plt.figure(figsize=(10, 7)) 
    plt.xlabel("Cluster Size")
    plt.ylabel("Distance")
    plt.show()
    
    #HAC clustering
    HClustering = AgglomerativeClustering(n_clusters , affinity="euclidean",linkage="ward").fit(final_matrix)
    HClustering.fit_predict(final_matrix)
    
    print(HClustering.labels_)
    clusters = collections.defaultdict(list)
    
    for i, label in enumerate(HClustering.labels_):
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

#print ("results :",clustering(3, final_matrix, df))
#for cluster in range(n_clusters):
#    print ("cluster ",cluster,":")
#    for i, gpe_sentences in enumerate(dict(clusters)[cluster]):
#        print("gpe of sentences : ", gpe_sentences)
#    nb_sentences = (gpe_sentences+1) / dfLen
#    print("nb sentences: ", nb_sentences)

#print (kmeans.labels_)
    
def key_words (cluster):
    #X = tfidf_vectorizer.fit_transform([cluster])
    #print("cluster", cluster)
#    tfidf_ = TfidfVectorizer()
## A mapping of terms to feature indices. => dictionnary
#    transformed = tfidf_.fit_transform([cluster])
#    print("key words tfidfVectorizer",transformed.get_feature_names())
   # print("key words tfidfVectorizer", tfidf_.get_feature_names())
    a = keywords(cluster, words = 5,scores = True, lemmatize = True)
    print("key words summarization gensim",a)
    #return tfidf_.get_feature_names()
    return a
    
    #a = keywords(text2)
    #password login
    #email
    #users
    
if __name__ =="__main__":
   
#k-means    
#    n_clusters = 5
#    new_clusters = clustering(n_clusters, final_matrix, df_orig)
#    for cluster in range(n_clusters):
#        tab_cluster = ""
#        print("cluster", cluster,"\n")
#        for i, sentenceIndex in enumerate(new_clusters[cluster]):
#            print("sentence ", i,":", df_orig[sentenceIndex],"\n")
#            tab_cluster = tab_cluster + str(df[sentenceIndex])
#        #key_words (tab_cluster)
#        key_words(tab_cluster)
            
   # print (process("easily connecting users to the portals"))

#HAC  
    #based on the dendrogram we have 5 clusetes 
    k =5 
    #build the model
    HClustering = AgglomerativeClustering(n_clusters=k , affinity="euclidean",linkage="ward")
    new_clusters = clustering_Hac(k, final_matrix, df_orig)
    for cluster in range(k):
        tab_cluster = ""
        print("cluster", cluster,"\n")
        for i, sentenceIndex in enumerate(new_clusters[cluster]):
            print("sentence ", i,":", df_orig[sentenceIndex],"\n")
            tab_cluster = tab_cluster + str(df[sentenceIndex])
        #key_words (tab_cluster)
        key_words(tab_cluster)

#    plt.title("Dendrograms")  
#    dend = shc.dendrogram(shc.Z)
    #cutting at y=3
#    cluster = fcluster(Z, 3, criterion='inconsistent', depth=2)

    # # Z = [[0,1,2],[3,4,5],[6,7,8]]
    # # Z[:2,1] is [1,4]
    # # Z[:2,:3] is [[0,1,2],[3,4,5]]
    # # Obtain node of maximum distance. Nodes denote the nodes of predicted clusters.
#    node, label = leaders(Z, cluster)
#    node_distance_dict = dict()
#    node_distance = []
#    for i in range(len(Z)):
#        for j in node:
#            if j in Z[i]:
#                node_distance.append(Z[i, 2])
#                node_distance_dict.update({j: Z[i, 2]})
#    max_node_distance = max(node_distance)
#    cluster_group = fcluster(Z, t=max_node_distance, criterion='distance')
#
#    if len(set(cluster_group)) < 2:
#        logging.warning(
#            'The number of clusters in clusterR is less than 2, which maybe leads to inappropriate hierarchical information of feature tree.'
#            'Try to adingjust topic words or final_t to obtian more clusters in cluster_group.')
#
#    print('\n============ Clustering Results ============')
#    #print('The selected inconsistency threshold:', incon_threshold)
#    print('Cluster Group:\n', cluster_group)
#    print('Predicted Clustering Labels:\n', cluster)
#    print('Label:\n', label)
#    print('\n')

