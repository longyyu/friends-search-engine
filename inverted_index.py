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

# Define a decorator to measure the execution times of dedicated methods
def measure_time(f):
  def timed(*args, **kw):
    time_start = time.time()
    result = f(*args, **kw)
    time_end = time.time()
    print("Executed {} in {:2.2f} sec. ".format(
      f.__name__, time_end - time_start
    ))
    return result
  return timed

class Indexes:
  def __init__(self, documents, stop_words, stem = False):
    self.documents = documents
    self.doc_count = len(documents)
    self.dict_df = None
    self.posting_df = None
    self.stop_list = stop_words
    # whether to stem the terms when tokenizing
    self.do_stem = stem

  def tokenize(self, document):
    # tokenize the given document
    # input: document - a string, the original document
    # output: a list of strings, terms from the original document
    #         stemmed if self.do_stem is True
    doc_tokens = word_tokenize(document.strip().lower())  
    if self.do_stem:
      doc_tokens = [porter.stem(term) for term in doc_tokens]
    else:
      doc_tokens = [term for term in doc_tokens]
    return(doc_tokens)

  @measure_time
  def generate_inverted_index(self):
    term_to_freq_pos_dict = dict()

    for doc_id, doc in enumerate(self.documents):
      if doc_id % 2000 == 0:
        print(f"    Processing document # {doc_id} ...")
      
      # tokenize the document
      doc_tokens = self.tokenize(doc)
      
      # update the posting
      for pos, term in enumerate(doc_tokens):
        if term in self.stop_list:
          continue
        if (doc_id, term) not in term_to_freq_pos_dict:
          term_to_freq_pos_dict[(doc_id, term)] = [0, []]
        # update freq and pos for current term
        term_to_freq_pos_dict[(doc_id, term)][0] += 1
        term_to_freq_pos_dict[(doc_id, term)][1].append(pos)
      
      # turn the posting into a data.frame
      self.posting_df = pd.DataFrame.from_dict(
        term_to_freq_pos_dict, orient = 'index', 
        columns = ["freq", "pos"]
      )
      docid_term = list(zip(*self.posting_df.index))
      self.posting_df['doc_id'] = docid_term[0]
      self.posting_df['term'] = docid_term[1]
      self.posting_df = self.posting_df.reset_index(drop = True)\
        [["doc_id", "term", "freq", "pos"]]\
        .sort_values(by = "doc_id")
      # create a dictionary base on posting
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

indexes = Indexes(
  documents = documents, 
  # documents = script_utterance.transcript.tolist(), 
  stop_words = stop_words,
  stem = True
)
indexes.generate_inverted_index()
print(indexes.dict_df.head())
print(indexes.posting_df.head(20))
