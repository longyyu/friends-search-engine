from nltk import word_tokenize
from nltk.stem import PorterStemmer
porter = PorterStemmer()
import pandas as pd
import numpy as np
import math
import os

from helper_func import *
from data_prep import script_utterance

# Purpose: This script #! add header
# Author: Yanyu Long
# Updated: Dec 11, 2020

class Indexes:
  def __init__(self, documents, stop_words, doc_id = None, stem = False):
    self.stop_list = stop_words # a list of stop words
    self.do_stem = stem # whether to stem the terms when tokenizing
    self.documents = documents
    self.doc_count = len(documents)
    self.doc_id = range(0, self.doc_count) if doc_id is None else doc_id
    self.doc_length = dict(zip(self.doc_id, [len(doc) for doc in documents]))
    self.avg_doc_length = np.mean(list(self.doc_length.values()))
    
    # self.doc_tokens: a dictionary that maps a document's ID to its tokens
    self.doc_tokens = read_dict("./data/doc_tokens.pkl") \
      if os.path.exists("./data/doc_tokens.pkl") else None
    if self.doc_tokens is None:
      self.tokenize_all_documents()
    
    # self.term_to_freq_pos: a dictionary that maps a tuple of (doc_id, term)
    # to a list of [term frequency (an integer), position (a integer list)]
    self.term_to_freq_pos = read_dict("./data/term_to_freq_pos.pkl") \
      if os.path.exists("./data/term_to_freq_pos.pkl") else None
    if self.term_to_freq_pos is None:
      self.generate_inverted_index()
    
    # self.doc_freq: a dictionary that maps term to its document frequency
    self.doc_freq = None
    self.compute_doc_freq()

  def tokenize(self, document, remove_stop_words = False):
    # tokenize the given document
    # inputs: 
    #   document - string, the original document
    #   remove_stop_words - boolean, whether to remove stop words from tokens
    # output: list of strings, terms from the original document
    #         stemmed if self.do_stem is True
    doc_tokens = word_tokenize(document.strip().lower())
    if remove_stop_words:
      doc_tokens = [term for term in doc_tokens if term not in self.stop_list]
    if self.do_stem:
      doc_tokens = [porter.stem(term) for term in doc_tokens]
    else:
      doc_tokens = [term for term in doc_tokens]
    return(doc_tokens)
  
  @measure_time
  def tokenize_all_documents(self):
    doc_tokens_list = []
    for doc in self.documents:
      doc_tokens_list.append(self.tokenize(doc))
    # update self.doc_tokens and save to disk
    self.doc_tokens = dict(zip(self.doc_id, doc_tokens_list))
    save_dict(self.doc_tokens, "./data/doc_tokens.pkl")

  @measure_time
  def generate_inverted_index(self):
    self.term_to_freq_pos = dict()
    for idx, doc_id in enumerate(self.documents):
      # print processing status every 2000 documents
      if idx % 2000 == 0:
        print("    Processing document # {:5d} (doc_id = {}) ...".\
          format(idx, doc_id)
        )
      # update posting (self.term_to_freq_pos)
      for pos, term in enumerate(self.doc_tokens[doc_id]):
        if term in self.stop_list:
          continue
        if (doc_id, term) not in self.term_to_freq_pos:
          self.term_to_freq_pos[(doc_id, term)] = [0, []]
        # update freq and pos for current term
        self.term_to_freq_pos[(doc_id, term)][0] += 1
        self.term_to_freq_pos[(doc_id, term)][1].append(pos)
    
    # save self.term_to_freq_pos and self.doc_freq to disk
    save_dict(self.term_to_freq_pos, "./data/term_to_freq_pos.pkl")
      
  @measure_time
  def compute_doc_freq(self):
    # turn the posting into a data.frame
    posting_df = pd.DataFrame.from_dict(
      self.term_to_freq_pos, orient = 'index', 
      columns = ["freq", "pos"]
    )
    docid_term = list(zip(*posting_df.index))
    posting_df['doc_id'] = docid_term[0]
    posting_df['term'] = docid_term[1]
    posting_df = posting_df.reset_index(drop = True)\
      [["doc_id", "term", "freq", "pos"]]\
      .sort_values(by = "doc_id")
    
    # create a dictionary base on posting_df
    dictionary_df = posting_df.pivot_table(
      index = 'term', values = ['doc_id', 'freq'],
      aggfunc = {'doc_id': 'count', 'freq': np.sum}
    ).reset_index()\
    .rename(columns = {'doc_id': 'doc_freq', 'freq': 'total_freq'})
    # update self.doc_freq
    self.doc_freq = dict(
      zip(dictionary_df['term'], dictionary_df['doc_freq'])
    )

  def bm25_score_term_doc(self, term, doc_id, k1 = 1.25, b = 0.75):
    df_term = self.doc_freq[term]
    tf_term_doc = self.term_to_freq_pos[(doc_id, term)][0]

    score_idf = math.log((self.doc_count - df_term + 0.5) / (df_term + 0.5))
    score_tf = ((k1 + 1) * tf_term_doc / 
                (k1 * (1 - b + b * self.doc_length[doc_id] / 
                self.avg_doc_length) + tf_term_doc))
    return(score_idf * score_tf)
  
  def bm25_score(self, query, doc_id_list = None, k1 = 1.25, b = 0.75):
    #! change `bm25_score` to general scoring
    #! add a map from string to ranking functions
    query_tokens = self.tokenize(query, remove_stop_words = True)
    if doc_id_list is None: 
      # if the user did not specify doc_id_list, will go through all documents
      doc_id_list = self.doc_id
    score_doc = [0] * len(doc_id_list)
    for i, doc_id in enumerate(doc_id_list):
      common_tokens = set(query_tokens) & set(self.doc_tokens[doc_id])
      for term in common_tokens:
        score_doc[i] += self.bm25_score_term_doc(term, doc_id) #! might want to specify k1 and b
    return(score_doc)

def get_retrieval_results(query, filter_by_character = "", num_results = 10):
  # filter documents to be queried
  if filter_by_character == "":
    query_doc_id = indexes.doc_id
  else:
    query_doc_id = script_utterance_tmp.loc[ #! update to all docs
      script_utterance_tmp.speakers == filter_by_character, "u_id"
    ].tolist()

  # rank the documents
  doc_score =  indexes.bm25_score(query = query, doc_id_list = query_doc_id)

  # organize the ranking results
  doc_score_df = pd.DataFrame(dict(doc_id = query_doc_id, score = doc_score)).\
    sort_values(by = "score", ascending = False).\
    reset_index(drop = True)
  return(doc_score_df.loc[0:(num_results - 1), 'doc_id'].tolist())

# -----------------------------------------------------------------------------
# import stop words
with open('./lemur-stopwords.txt', 'r',
          encoding = "UTF-8") as f:
  stop_words = [line.strip() for line in f]
  f.close()

# set up documents and doc_id
script_utterance_tmp = script_utterance.iloc[:4001, ] #! update to all docs
documents = script_utterance_tmp.transcript.tolist() #! update to all docs
doc_id = script_utterance_tmp.u_id.tolist() #! update to all docs

# build inverted index
indexes = Indexes(
  documents = documents, 
  doc_id = doc_id, 
  stop_words = stop_words,
  stem = False
)

if __name__ == "__main__":
  print(get_retrieval_results(
    query = "you're going out with the guy",
    filter_by_character = "Joey Tribbiani"
  ))
