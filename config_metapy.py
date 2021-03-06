import os
from metapy import metapy
from data_prep import script_utterance

# Metapy Settings
#  - Creating config files, DAT files, inv_idx
#  - function definition: get_retrieval_results
# Term Project, SI650, F20
# Author: Yanyu Long, longyyu@umich.edu
# Updated: Dec 13, 2020

# create config file ----------------------------------------------------------
config_file = "./data/friends-config.toml"

if not os.path.exists(config_file):
  config_content = """prefix = "./data"
dataset = "friends"
corpus = "tutorial.toml"

index = "./data/friends-idx"

query-judgements = "./data/friends-qrels.txt"
stop-words = "./data/lemur-stopwords.txt"

[[analyzers]]
method = "ngram-word"
ngram = 1
filter = "default-unigram-chain"
"""

  with open(config_file, 'w') as f:
    _ = f.write(config_content)
  f.close()
  
  with open('./data/friends/tutorial.toml', 'w') as f:
    _ = f.write('type = "line-corpus"\n')
    _ = f.write('store-full-text = true\n')
  f.close()
else:
  print(f"{config_file} already exists.")

# Create .dat file ------------------------------------------------------------
output_dir = "./data/friends/"

if not os.path.exists(f"{output_dir}friends.dat"):
  with open(f"{output_dir}friends.dat", 
            'w', encoding = "UTF-8") as f:
    for line in script_utterance.transcript:
      _ = f.write(f"{line}\n")
    f.close()

# Make inverted index ---------------------------------------------------------
inv_idx = metapy.index.make_inverted_index(config_file)

# function: get_retrieval_results ---------------------------------------------
def get_retrieval_results(query_content, ranker, inv_idx, df_uid, 
                          num_results = 10,
                          return_type = "u_id"):
  # for the given query, return retrieved documents using metapy's framework
  # Inputs: query_content - content of the query
  #         ranker - the ranker to score with
  #         inv_idx - a metapy.metapy.index.InvertedIndex object
  #         num_results - number of documents to retrieve for each query
  # Outputs: retrieval_results - a list of tuples <query_id, doc_id> 
  #         representing the documents that scored highest for each query, 
  #         sorted in descending order of score
  query = metapy.index.Document()
  query.content(query_content)
  results = ranker.score(inv_idx, query, num_results) # ranker is called here
  if return_type == "u_id":
      return([df_uid.loc[x[0], 'u_id'] for x in results])
  else:
      return([x[0] for x in results])


if __name__ == "__main__":
  import pandas as pd
  from data_prep import get_script_with_uid, query_list
  ranker =  metapy.index.OkapiBM25(k1 = 1.2, b = 0.75, k3 = 500)
  
  # baseline model evaluation using metapy's IREval -------------------------
  num_testing_queries = len(query_list)  
  df_baseline_eval = pd.DataFrame(
      index = pd.Index(list(range(num_testing_queries)) + ["Mean"]), 
      columns = ["AP", "NDCG"]
  )

  ev = metapy.index.IREval(config_file)
  num_results = 10
  for query_num, query_content in enumerate(query_list):
    query = metapy.index.Document()
    query.content(query_content)
    results = ranker.score(inv_idx, query, num_results)     

    df_baseline_eval.iloc[query_num] = dict(
      AP = ev.avg_p(results, query_num, num_results), 
      NDCG = ev.ndcg(results, query_num, num_results)
    )
  # calculate average AP and NDCG
  df_baseline_eval.iloc[num_testing_queries] = dict(
    AP = ev.map(), NDCG = df_baseline_eval.NDCG.mean()
  )
  print(df_baseline_eval)
