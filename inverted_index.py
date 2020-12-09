from nltk import word_tokenize
from nltk.stem import PorterStemmer
porter = PorterStemmer()
import pandas as pd
import numpy as np
import time

from data_prep import script_utterance

# Purpose: This script #! add header
# Author: Yanyu Long
# Updated: Dec 10, 2020

class Indexes:
  def __init__(self, documents, stop_words):
    self.documents = documents
    self.doc_count = len(documents)
    self.dict_df = None
    self.posting_df = None
    self.stop_list = stop_words

  def generate_inverted_index(self, doStem = False):
    term_to_freq_dict = dict()
    term_to_pos_dict = dict()

    for doc_id, doc in enumerate(self.documents):
      # tokenize the document
      doc_tokens = word_tokenize(doc.strip().lower())  
      if doStem:
        doc_tokens = [porter.stem(term) for term in doc_tokens]
      else:
        doc_tokens = [term for term in doc_tokens]
      
      # update the posting
      for pos, term in enumerate(doc_tokens):
        if term in self.stop_list:
          continue
        if (doc_id, term) not in term_to_freq_dict:
          term_to_freq_dict[(doc_id, term)] = 0
          term_to_pos_dict[(doc_id, term)] = []
        # update freq and pos for current term
        term_to_freq_dict[(doc_id, term)] += 1
        term_to_pos_dict[(doc_id, term)].append(pos)
      
      # turn the posting into a data.frame
      self.posting_df = pd.merge(
        pd.DataFrame.from_dict(term_to_freq_dict, orient = 'index',
                               columns = ["freq"]), 
        pd.DataFrame([term_to_pos_dict], 
                     index = pd.Index(['pos'])).transpose(),
        left_index = True, right_index = True
      )
      docid_term = list(zip(*self.posting_df.index))
      self.posting_df['doc_id'] = docid_term[0]
      self.posting_df['term'] = docid_term[1]
      self.posting_df = self.posting_df.reset_index(drop = True)\
        [["doc_id", "term", "freq", "pos"]]\
        .sort_values(by = "doc_id")

      self.dict_df = self.posting_df.pivot_table(
        index = 'term', values = ['doc_id', 'freq'],
        aggfunc = {'doc_id': 'count', 'freq': np.sum}
      ).reset_index()

# -----------------------------------------------------------------------------
documents = [
   "Human machine interface, for lab abc computer applications",
   "A computer survey of user opinion of computer system response time",
   "The EPS user interface! management system",
   "System and human system. engineering testing of EPS",
   "Relation of user perceived response time to error measurement",
   "The generation of random binary unordered trees",
   "The intersection graph of, paths in trees",
   "Graph minors IV Widths of trees and well quasi ordering",
   "Graph minors A survey"
]

# import stop words
with open('./lemur-stopwords.txt', 'r',
          encoding = "UTF-8") as f:
  stop_words = [line.strip() for line in f]
  f.close()

time_start = time.time()
indexes = Indexes(documents = script_utterance.transcript.tolist(), 
                  stop_words = stop_words)
indexes.generate_inverted_index(doStem = True)
time_elapsed = time.time() - time_start
print("Finished in {:5.2f} seconds".format(time_elapsed))