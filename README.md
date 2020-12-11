# Search Engine for Friends Quotes  

Visit the Friends search engine [web app](http://longyyu.pythonanywhere.com/).  

## Source files  

*  data/script_id_speaker_10seasons.tsv stores the Friends corpus data
*  data_prep.py reads in data from the above tsv file as pd.DataFrame object
*  web_ui.py is the user interface (implemented using `Flask`) for the web app
*  templates/ stores the html templates for the web app
*  static/ stores the static resources that will be utilized by the web app  
  

## Baseline Model  

The baseline model adopts a BM25 ranker implemented using `metapy`.  
The following files and directories are for the metapy baseline model:  
*  config_metapy.py
*  ./data/friends/
*  ./data/friends-config.toml
*  ./data/lemur-stopwords.txt
  
## My own ranking function  

Still under development ...
