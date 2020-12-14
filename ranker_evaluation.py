import pandas as pd
import numpy as np
import math

from data_prep import query_list, query_relevance, uid_to_rowidx

# Purpose: This script defines functions that are used to evaluate a 
#          information retrieval model's performance. 
# Author: Yanyu Long
# Updated: Dec 14, 2020

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


if __name__ == "__main__":
  NUM_RESULT = 10

  # # evaluate the baseline model ----------------------------
  # from metapy import metapy
  # from config_metapy import inv_idx, \
  #   get_retrieval_results as get_retrieval_results_metapy
  # from data_prep import script_utterance, query_list
  # ranker =  metapy.index.OkapiBM25(k1 = 1.2, b = 0.75, k3 = 500)  
  # # retrieve documents
  # query_result = pd.DataFrame()
  # for q_id, query in enumerate(query_list):
  #   query_result = query_result.append(
  #     pd.DataFrame(dict(
  #       query_id = q_id, 
  #       doc_id = get_retrieval_results_metapy(
  #         query, ranker, inv_idx, script_utterance, 
  #         num_results = NUM_RESULT,
  #         return_type = "row_idx"
  #       )
  #     )), 
  #     ignore_index = True
  #   )
  # # evaluate ranker performance
  # baseline_eval = evaluate_query_result(query_result)
  
  # baseline_eval = baseline_eval.append(pd.DataFrame.from_records(
  #   dict(baseline_eval[["ap", "ndcg"]].mean()), index = pd.Index(["Mean"])
  # ))
  # print(baseline_eval)

  # evaluate other ranking functions ---------------------------
  from inverted_index import get_retrieval_results
  from itertools import chain

  def evaluate_ranker(ranker, **kwargs):
    # given ranker name and parameters (specified by **kwargs),
    # evaluate the ranker's performance on each testing query using AP and 
    # NDCG, and returns a data frame containing the evaluation results
    
    # retrieve documents
    query_result = pd.DataFrame()
    for q_id, query in enumerate(query_list):
      query_result = query_result.append(
        pd.DataFrame(dict(
          query_id = q_id, 
          doc_id = get_retrieval_results(
            query = query, ranker = ranker, num_results = NUM_RESULT, **kwargs
          )
        )), 
        ignore_index = True
      )
    # transform utterance ID into document row index
    query_result['doc_id'] = query_result['doc_id'].apply(
      lambda x: uid_to_rowidx[x]
    )
    # evaluate ranker performance
    ranker_eval = evaluate_query_result(query_result)
    ranker_eval_avg = dict(ranker_eval[["ap", "ndcg"]].mean())
    # add ranker and parameters to the dictionary
    params = ', '.join([
      "{}={}".format(key, val) for key, val in kwargs.items()
    ])
    ranker_eval_avg.update(dict(ranker = ranker, params = params))
    return(pd.DataFrame.from_records(ranker_eval_avg, index = pd.Index([0])))
  
  # # bm25 -------------------------------------------------------------------
  # k1_val = np.arange(0.4, 2.0, 0.2).tolist()
  # b_val = np.arange(0.6, 0.9, 0.05).tolist()
  # rankers = pd.DataFrame(dict(
  #   ranker = "bm25",
  #   k1 = list(chain.from_iterable(
  #     [[val] * len(b_val) for val in k1_val]
  #   )),
  #   b = b_val * len(k1_val)
  # ))
  # rankers_eval = pd.concat(
  #   list(rankers.apply(
  #     lambda row: evaluate_ranker(
  #       ranker = row["ranker"], k1 = row["k1"], b = row["b"]
  #     ), axis = 1
  #   )), 
  #   ignore_index = True
  # )
  # rankers_eval[["ranker", "params", "ap", "ndcg"]].to_csv(
  #   "./data/ranker_evaluation_bm25.csv", index = False
  # )
  # print(rankers_eval.sort_values(by = "ap", ascending = False).head(10))

  # # bm25_v1 ---------------------------------------------------------------
  # k1_val = np.arange(0.4, 2.0, 0.2).tolist()
  # b_val = np.arange(0.6, 0.9, 0.05).tolist()
  # rankers = pd.DataFrame(dict(
  #   ranker = "bm25_v1",
  #   k1 = list(chain.from_iterable(
  #     [[val] * len(b_val) for val in k1_val]
  #   )),
  #   b = b_val * len(k1_val)
  # ))
  # rankers_eval = pd.concat(
  #   list(rankers.apply(
  #     lambda row: evaluate_ranker(
  #       ranker = row["ranker"], k1 = row["k1"], b = row["b"]
  #     ), axis = 1
  #   )), 
  #   ignore_index = True
  # )
  # rankers_eval[["ranker", "params", "ap", "ndcg"]].to_csv(
  #   "./data/ranker_evaluation_bm25_v1.csv", index = False
  # )
  # print(rankers_eval.sort_values(by = "ap", ascending = False).head(10))

  # # pivoted length ------------------------------------------------------
  # rankers = pd.DataFrame(dict(
  #   ranker = "piv", b = np.arange(0.05, 1.00, 0.05).tolist()
  # ))
  # rankers_eval = pd.concat(
  #   list(rankers.apply(
  #     lambda row: evaluate_ranker(
  #       ranker = row["ranker"], b = row["b"]
  #     ), axis = 1
  #   )), 
  #   ignore_index = True
  # )
  # rankers_eval[["ranker", "params", "ap", "ndcg"]].to_csv(
  #   "./data/ranker_evaluation_piv.csv", index = False
  # )
  # print(rankers_eval.sort_values(by = "ap", ascending = False).head(10))

  # # tsl ---------------------------------------------------------------
  # rankers = pd.DataFrame(dict(
  #   ranker = "tsl", mu = np.arange(2200, 4200, 200).tolist()
  # ))
  # rankers_eval = pd.concat(
  #   list(rankers.apply(
  #     lambda row: evaluate_ranker(
  #       ranker = row["ranker"], mu = row["mu"]
  #     ), axis = 1
  #   )), 
  #   ignore_index = True
  # )
  # rankers_eval[["ranker", "params", "ap", "ndcg"]].to_csv(
  #   "./data/ranker_evaluation_tsl.csv", index = False
  # )
  # print(rankers_eval.sort_values(by = "ap", ascending = False).head(10))

  # others: ES & F2EXP ----------------------------------------------------
  print(
    pd.concat(
      [evaluate_ranker(ranker = "es"), evaluate_ranker(ranker = "f2exp")],
      ignore_index = True
    )[["ranker", "params", "ap", "ndcg"]]
  )
