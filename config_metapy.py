"""
Metapy Settings
 - Creating config files, DAT files, inv_idx
 - function definition: get_retrieval_results
Term Project, SI650, F20
Author: Yanyu Long, longyyu@umich.edu
Updated: Nov 27, 2020
"""

import os
from metapy import metapy

# create config file ----------------------------------------------------------
config_file = "friends-config.toml"
if not os.path.exists(config_file):
    config_content = """
prefix = "."

dataset = "friends"
corpus = "tutorial.toml"

index = "friends-idx"

query-judgements = "friends/friends-qrels.txt"

stop-words = "lemur-stopwords.txt"

[[analyzers]]
method = "ngram-word"
ngram = 1
filter = "default-unigram-chain"
"""
    with open(config_file, 'w') as f:
        _ = f.write(config_content)
    f.close()
    
    with open('friends/tutorial.toml', 'w') as f:
        _ = f.write('type = "line-corpus"\n')
        _ = f.write('store-full-text = true\n')
    f.close()
else:
    print(f"{config_file} already exists.")

# Create .dat file ------------------------------------------------------------
from data_prep import script_utterance
output_dir = "./friends/"

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
