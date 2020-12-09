"""
Data Preparation
 - Read JSON file as pd.DataFrame, store as TSV
 - function definition: get_script_with_u_id
Term Project, SI650, F20
Author: Yanyu Long, longyyu@umich.edu
Updated: Nov 27, 2020
"""
# 79: -------------------------------------------------------------------------
import os
import json
import pandas as pd
import re

json_dir = "./data/json/"
json_file = json_dir + "friends_season_{:02d}.json"
output_dir = "./data/"

# Data preparation ------------------------------------------------------------
def get_script_utterance(season_id):
    def get_utterance_count(season):
        num_utterances = 0
        for episode in season['episodes']:
            for scene in episode['scenes']:
                for utterance in scene['utterances']:
                    if utterance['tokens']:
                        num_utterances += 1
        return(num_utterances)
    
    season = json.load(open(json_file.format(season_id)))    
    df_all_transcript = pd.DataFrame(
        index=range(get_utterance_count(season)), 
        columns=['u_id', 'speakers', 'transcript']
    )
    idx = 0          
    for episode in season['episodes']:
        for scene in episode['scenes']:
            for utterance in scene['utterances']:
                if utterance['tokens']:
                    df_all_transcript.iloc[idx] = dict(
                        u_id = utterance['utterance_id'],
                        speakers = ', '.join(utterance['speakers']),
                        transcript = utterance['transcript']
                    )
                    idx += 1
    return(df_all_transcript)

if not os.path.exists(f"{output_dir}script_id_speaker_10seasons.tsv"):
    script_utterance = pd.DataFrame(columns=['u_id', 'speakers', 'transcript'])
    for season_id in range(1, 11):
        script_utterance = script_utterance.append(
            get_script_utterance(season_id), 
            ignore_index = True
        )
    script_utterance.to_csv(
        f"{output_dir}script_id_speaker_10seasons.tsv",
        sep = '\t', index = False
    )
else:
    script_utterance = pd.read_csv(
        f"{output_dir}script_id_speaker_10seasons.tsv",
        sep = '\t', header = 0
    )

# function: pretty_cid --------------------------------------------------------
def pretty_cid(cid):
  cid = re.sub("_", " ",
    re.sub("C", "Scene", 
      re.sub("E", "Episode", 
        re.sub("S", "Season", cid.upper())
      )
    )
  )
  return(cid)

# function: get_script_with_u_id ----------------------------------------------
def get_script_with_u_id(df, u_id, plus_minus = 0, output_format = "terminal"):
  #! add header here
  #! might add output_format = ["terminal", "html"]
  if output_format == "terminal":
    sym_newline = "\n"
    sym_red = "\x1b[1;31;47m"
    sym_normal = "\x1b[0m"
  else:
    sym_newline = "<br>"
    sym_red = "<span style='color:IndianRed'>"
    sym_normal = "</span>"
    
  if plus_minus <= 0:
      df_target = df.loc[df.u_id == u_id].reset_index(drop = True)
      script = "Episode: {}{}[{}] {}{}".format(
          df_target.u_id.str.extract(r"(.*)_c").loc[0, 0],
          sym_newline,
          df_target.loc[0, "speakers"],
          df_target.loc[0, "transcript"],
          sym_newline
      )
  else:
      cid = pd.Series(u_id).str.extract(r"(.*)_u").loc[0, 0]
      row_idx = df.loc[(df.u_id == u_id)].index.tolist()[0]
      target_uid = df.loc[range(row_idx-plus_minus, 
                                row_idx+plus_minus + 1), 'u_id']
      target_uid = target_uid[target_uid.str.contains(cid)]

      # initialize script with utterance/scene ID
      if output_format == "terminal":
        script = f"{u_id} ({row_idx})\n"
      else:
        script = "<span style='background-color: WhiteSmoke;'>" + \
                 pretty_cid(cid) + "</span><br>"

      for cur_uid in target_uid:
          df_target = df.loc[df.u_id == cur_uid].\
              reset_index(drop = True)
          if cur_uid == u_id:
              script += "{}[{}] {}{}{}".format(
                  sym_red,
                  df_target.loc[0, "speakers"],
                  df_target.loc[0, "transcript"],
                  sym_normal,
                  sym_newline
              )
          elif len(df_target) > 0:
              script += "[{}] {}{}".format(
                  df_target.loc[0, "speakers"],
                  df_target.loc[0, "transcript"],
                  sym_newline
              )
  return(script)
