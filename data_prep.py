import os
import json
import pandas as pd
import re

# Data Preparation
#  - Read JSON file as pd.DataFrame, store as TSV
#  - function definition: get_script_with_uid, get_episode_with_uid
# Term Project, SI650, F20
# Author: Yanyu Long, longyyu@umich.edu
# Updated: Dec 12, 2020

json_dir = "./data/json/"
json_file = json_dir + "friends_season_{:02d}.json"
output_dir = "./data/"

# Data preparation ------------------------------------------------------------
def get_script_utterance(season_id):
  # read script data from json files and store as pd.DataFrame object
  # input - season_id: an integer

  def get_utterance_count(season):
    # a helper function that returns the number of utterances in a season
    # this is used to specify number of rows in data frame df_all_transcript 
    # to avoid appending rows, which will expedite the outer function
    # input - season: json format data
    num_utterances = 0
    for episode in season['episodes']:
      for scene in episode['scenes']:
        for utterance in scene['utterances']:
          if utterance['tokens']:
            num_utterances += 1
    return(num_utterances)
    
  season = json.load(open(json_file.format(season_id)))
  df_all_transcript = pd.DataFrame(
    index = range(get_utterance_count(season)), 
    columns = ['u_id', 'speakers', 'transcript']
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

# if the TSV format data already exists on disk, then read from TSV
# otherwise read from JSON format data and store as TSV
if os.path.exists(f"{output_dir}script_id_speaker_10seasons.tsv"):
  script_utterance = pd.read_csv(
    f"{output_dir}script_id_speaker_10seasons.tsv",
    sep = '\t', header = 0
  )
else:
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

# function: pretty_cid --------------------------------------------------------
def pretty_cid(cid):
  # a helper function that prints scene id (cid) in a nice format
  # input - cid: a string in the format of "s01_e01_c01"
  cid = re.sub("_", " ",
    re.sub("C", "Scene", 
      re.sub("E", "Episode", 
        re.sub("S", "Season", cid.upper())
      )
    )
  )
  return(cid)

# function: get_script_with_uid ----------------------------------------------
def get_script_with_uid(df, u_id, plus_minus = 0, output_format = "terminal"):
  #! add header here
  # inputs:
  #   df: 
  #   ... 
  #   output_format: one of ["terminal", "html"]
  if output_format == "terminal":
    sym_newline = "\n"
    sym_red = "\x1b[1;31;47m"
    sym_normal = "\x1b[0m"
  else:
    sym_newline = "<br>"
    sym_red = "<span style='color:IndianRed'>"
    sym_normal = "</span>"
  
  if plus_minus <= 0:
    df_target = df.loc[df.u_id == u_id].reset_index()
    script = "{} ({}){}[{}] {}{}".format(
      u_id,
      df_target.loc[0, "index"],
      sym_newline,
      df_target.loc[0, "speakers"],
      df_target.loc[0, "transcript"],
      sym_newline
    )
  else:
    cid = pd.Series(u_id).str.extract(r"(.*)_u").loc[0, 0]
    row_idx = df.loc[(df.u_id == u_id)].index.tolist()[0]
    df_target = df.loc[range(max(0, row_idx - plus_minus), 
                             row_idx+plus_minus + 1)]
    df_target = df_target.loc[df_target.u_id.str.contains(cid)]\
                         .reset_index(drop = True)

    # initialize script with utterance/scene ID
    if output_format == "terminal":
      script = f"{u_id} ({row_idx})\n"
    else:
      script = "<span style='background-color: WhiteSmoke;'>" + \
               "<a href='/script/{}'>{}</a>".format(u_id, pretty_cid(cid)) + \
               "</span><br>"

    # for cur_uid in target_uid:
    for i in range(len(df_target)):
      if df_target.loc[i, "u_id"] == u_id: 
        # highlight the target utterance in red
        script += "{}[{}] {}{}{}".format(
          sym_red,
          df_target.loc[i, "speakers"],
          df_target.loc[i, "transcript"],
          sym_normal,
          sym_newline
        )
      else:
        script += "[{}] {}{}".format(
          df_target.loc[i, "speakers"],
          df_target.loc[i, "transcript"],
          sym_newline
        )
  return(script)

# function: get_episode_with_uid ----------------------------------------------
def get_episode_with_uid(df, uid):
  #! add header

  sym_newline = "<br>"
  def format_scene_script(cid, df_scene):
    cid_fmt = "{}<span style='background-color: WhiteSmoke; "\
              "font-size: 18px;'>{}</span>{}{}"
    script = cid_fmt.format(
      sym_newline,
      "Scene {:2d}\n".format(int(
        re.compile("c(.*)").findall(cid)[0]
      )),
      sym_newline, sym_newline
    )
    for row_idx in range(len(df_scene)):
      script += "[{}]  {}{}".format(
          df_scene.loc[row_idx, "speakers"],
          df_scene.loc[row_idx, "transcript"],
          sym_newline
        )
    return(script)
  
  # extract episode id from utterance id
  eid = re.compile("s[0-9]{2}_e[0-9]{2}").findall(uid)[0]
  # extract all rows of that episode from df
  df_target = df.loc[df.u_id.str.match(eid), ].reset_index(drop = True)
  # extract scene id (cid) of all scenes
  df_target['c_id'] = df_target.u_id.str.extract(r"(.*)_u")
  cid_list = df_target.c_id.unique()
  # format and concatenate script from each scene
  episode = sym_newline.join([
    format_scene_script(
      cid, df_target.loc[df_target.c_id == cid].reset_index(drop = True)
    ) for cid in cid_list
  ])

  return(episode)

# generate a list of characters sorted in descending order of 
# total utterances across all ten seasons
character_list = script_utterance.pivot_table(
    index = "speakers", values = "u_id", aggfunc = "count"
  ).sort_values(by = "u_id", ascending = False)\
    .index.tolist()
character_list.remove("#ALL#")

if __name__ == "__main__":
  # print out the character list in descending order of their 
  # total utterances across ten seasons
  print(
    [("", "Select a character")] + \
      [(name, name.split(" ")[0]) for name in character_list[:20]]
  )
