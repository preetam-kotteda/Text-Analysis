#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
import math
from operator import itemgetter


# In[4]:


#importing the file
raw = open('kgraph.csv', 'r').read()


# In[5]:


type(raw)


# In[6]:


#removing punctuation
import re
raw_parsed = re.sub("[^-9A-Za-z ]", "" , raw)


# In[8]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.add('.')


# In[26]:


#tokenization
from nltk import tokenize
tokens = nltk.word_tokenize(raw_parsed)
words = [w.lower() for w in tokens]
total_words = [w for w in words if not w.lower() in stop_words]


# In[27]:


tokens


# In[28]:


# normalization
words = [w.lower() for w in tokens]
vocab = sorted(set(words))


# In[29]:


# stemming 
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()


# In[30]:


[porter.stem(t) for t in tokens]


# In[31]:


[lancaster.stem(t) for t in tokens]


# In[32]:


#Lemmatization
wnl = nltk.WordNetLemmatizer()
tokens_lemmatized = [wnl.lemmatize(t) for t in tokens]


# In[33]:


#pos-tagging
tokens_tagged = nltk.pos_tag(tokens)


# In[34]:


tokens_lemmatized_tagged = nltk.pos_tag(tokens_lemmatized)
tokens_lemmatized_tagged


# In[35]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.add('.')
filtered_sentence = [w for w in words if not w.lower() in stop_words]


# In[36]:


words_text = nltk.Text(filtered_sentence)
fdist = nltk.FreqDist(words_text)


# In[37]:


fdist.plot(50, cumulative=True)


# In[38]:


fdist.items()


# In[39]:


fdist.most_common(5)


# In[40]:


total_word_length = len(total_words)
print(total_word_length)


# In[41]:



tf_score = {}
for each_word in total_words:
    each_word = each_word.replace('.','')
    if each_word not in stop_words:
        if each_word in tf_score:
            tf_score[each_word] += 1
        else:
            tf_score[each_word] = 1

# Dividing by total_word_length for each dictionary element
tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
# print(tf_score)

total_sentences = tokenize.sent_tokenize(raw)
total_sent_len = len(total_sentences)
# print(total_sent_len)

def check_sent(word, sentences): 
    final = [all([w in x for w in word]) for x in sentences] 
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))

idf_score = {}
for each_word in total_words:
    each_word = each_word.replace('.','')
    if each_word not in stop_words:
        if each_word in idf_score:
            idf_score[each_word] = check_sent(each_word, total_sentences)
        else:
            idf_score[each_word] = 1

# Performing a log and divide
idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())

# print(idf_score)

tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
# print(tf_idf_score)

def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n]) 
    return result

print(get_top_n(tf_idf_score,50))


# In[42]:


import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt


# In[211]:


def getSentences(text):
    nlp = English()
    nlp.add_pipe('sentencizer')
    document = nlp(text)
    return [sent.strip() for sent in document.sents]

def printToken(token):
    print(token.text, "->", token.dep_)

def appendChunk(original, chunk):
    return original + ' ' + chunk

def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)

def isConstructionCandidate(token):
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)

def processSubjectObjectPairs(tokens):
    subject = ''
    object = ''
    relation = ''
    subjectConstruction = ''
    objectConstruction = ''
    for token in tokens:
        printToken(token)
        if "punct" in token.dep_:
            continue
        if isRelationCandidate(token):
            relation = appendChunk(relation, token.lemma_)
        if isConstructionCandidate(token):
            if subjectConstruction:
                subjectConstruction = appendChunk(subjectConstruction, token.text)
            if objectConstruction:
                objectConstruction = appendChunk(objectConstruction, token.text)
        if "subj" in token.dep_:
            subject = appendChunk(subject, token.text)
            subject = appendChunk(subjectConstruction, subject)
            subjectConstruction = ''
        if "obj" in token.dep_:
            object = appendChunk(object, token.text)
            object = appendChunk(objectConstruction, object)
            objectConstruction = ''
    print (subject.strip(), ",", relation.strip(), ",", object.strip())
    return (subject.strip(), relation.strip(), object.strip())

def processSentence(sentence):
    tokens = nlp_model(sentence)
    return processSubjectObjectPairs(tokens)

def printGraph(triples):
    G = nx.Graph()
    for triple in triples:
        G.add_node(triple[0])
        G.add_node(triple[1])
        G.add_node(triple[2])
        G.add_edge(triple[0], triple[1])
        G.add_edge(triple[1], triple[2])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(25, 25))
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='skyblue', alpha=0.9,
            labels={node: node for node in G.nodes()})
    plt.axis('off')
    plt.show()


# In[205]:


nlp = English()
nlp.add_pipe('sentencizer')


def split_in_sentences(text):
    doc = nlp(text)
    return [str(sent).strip() for sent in doc.sents]


# In[206]:


sentences = split_in_sentences(raw)


# In[207]:


nlp_model = spacy.load('en_core_web_sm')


# In[208]:


triples = []


# In[209]:


for sentence in sentences:
    triples.append(processSentence(sentence))


# In[212]:


printGraph(triples)


# In[ ]:




