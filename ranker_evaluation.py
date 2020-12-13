import pandas as pd
import numpy as np
import math

from data_prep import query_list, query_relevance

# Purpose: This script defines functions that are used to evaluate a 
#          information retrieval model's performance. 
# Author: Yanyu Long
# Updated: Dec 13, 2020

def calc_avg_precision(relevance_score):
  # input - relevance_score: an iterable object with numeric values
  df = pd.DataFrame(dict(
    relevance_binary = [1 if val > 0 else 0 for val in relevance_score]
  ))
  # calculate average precision
  df = df.loc[df.relevance_binary == 1] # filter the relevant documents
  df = df.assign( 
    precision = range(1, len(df) + 1) / (df.index + 1)
    # <num of retrieved, relevant docs> / <num of retrieved docs>
  )
  return(df.precision.mean())

def calc_dcg(relevance_score):
  # input - relevance_score: an iterable object with numeric values
  
  # formula 1 : [(rel / math.log(i + 2, 2)) \
  #              for i, rel in enumerate(relevance_score)]
  #     widely adopted; only the first element is not discounted
  # formula 2: [rel if i == 0 else (rel / math.log(i + 1, 2)) \
  #              for i, rel in enumerate(relevance_score)]
  #     both the first two elements are not discounted
  return(
    np.sum([rel if i == 0 else (rel / math.log(i + 1, 2)) \
            for i, rel in enumerate(relevance_score)])
  )
  
def calc_ndcg(rel_retrieved, rel_ideal):
  # inputs:
  #   rel_retrieved: an iterable object documenting the relevance score
  #     for retrieved documents
  #   rel_ideal: an iterable object documenting the ideal relevance score
  dcg = calc_dcg(rel_retrieved)
  idcg = calc_dcg(rel_ideal)
  return(dcg / idcg)

def evaluate_query_result(query_result):
  # evaluate the ranker's performance using AP and NDCG
  # inputs: query_result, a pd.DataFrame object with two columns: 
  #         ["query_id", "doc_id"]
  # output: a pd.DataFrame object with columns ["query_id", "ap", "ndcg"]

  # infer the number of documents retrieved for each query
  num_result = len(query_result.loc[query_result.query_id == 0])
  # merge with query_relevance for get relevance score
  query_result = query_result.merge(
    query_relevance, how = "left", on = ["query_id", "doc_id"]
  )
  # fill missing values (query-doc pairs that were not annotated) with zero
  query_result = query_result.fillna({'relevance': 0})
  # evaluate query results using AP and NDCG
  evaluation_result = pd.DataFrame(dict(
    query_id = range(len(query_list)),
    ap = [calc_avg_precision(
      query_result.loc[query_result.query_id == q_id, "relevance"])
      for q_id in range(len(query_list))
    ], 
    ndcg = [calc_ndcg(
      rel_retrieved = query_result.loc[
        query_result.query_id == q_id, "relevance"
      ],
      rel_ideal = query_relevance.loc[
        query_relevance.query_id == q_id, "relevance"
      ].sort_values(ascending = False).head(num_result)
      ) for q_id in range(len(query_list))
    ]
  ))
  # evaluation_result = evaluation_result.append(dict(
  #   query_id = None, 
  #   ap = evaluation_result.ap.mean(),
  #   ndcg = evaluation_result.ndcg.mean()
  # ), ignore_index = True)
  return(evaluation_result)