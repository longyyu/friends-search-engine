# Search Engine for Friends Quotes  

A vertical search engine for scripts of FRIENDS, the popular American TV sitcom.  
**Visit the web app [here](http://longyyu.pythonanywhere.com/)!**&nbsp;&nbsp;
*(link to web app last updated on Dec 15, 2020)*  

## About  

This is the term project for SI650 / EECS549 at the University of Michigan.  
The FRIENDS corpus data comes from github repository [emorynlp/character-mining](https://github.com/emorynlp/character-mining).

## Setup  
***Note: this project is written in Python3 (3.7).*** 

* clone this repository  
* install the required packages: `pip3 install -r requirements.txt`
* download NLTK's sentence tokenizer `Punkt` by running `import nltk; nltk.download('punkt')` in Python

## Usage  
* change directory to the project's root folder
* run `python -m web_ui`

## Source files  

![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+)
***Python scripts***  
├── config_metapy.py *# set up the baseline model (metapy)*  
├── data_prep.py *# read in and pre-process data*  
├── helper_func.py *# defines helper functions*  
├── inverted_index.py *# defines class Indexes, which builds up inverted index and ranks documents*  
├── ranker_evaluation.py *# evaluates ranker performance using AP and NDCG*  
├── web_ui.py *# defines the flask framework of the web app*  
<br>
![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+)
***data files***  
├── **data**  
│   ├── json.tar.gz *# original FRIENDS corpus data in JSON format, archived*  
│   ├── script_id_speaker_10seasons.tsv *# pre-processed FRIENDS corpus data*  
│   ├── lemur-stopwords.txt *# the baseline stop words data*  
│   ├── stopwords.txt *# stop words data, personal pronouns removed*  
│   ├── friends-qrels.txt *# query judgement data for ranker evaluation purpose*  
│   ├── friends-queries.txt *# testing queries*  
│   ├── friends-config.toml *# metapy config file*  
│   ├── friends/... *# data input for metapy*  
│   ├── friends-idx/... *# inverted index built by metapy*  
│   ├── term_to_freq_pos.pkl *# temporary data saved locally to expedite application setup*  
│   ├── corpus_term_freq.pkl *# same as above*  
│   └── doc_tokens.pkl *# same as above*  
<br>
![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+)
***files for the web app's user interface***  
├── **static** *# static resources*  
│   ├── FriendsLogo.png  
│   ├── blogpost.html  
│   ├── mystyle.css  
│   └── sample-search/...  
└── **templates** *# template pages*  
&nbsp;&nbsp;&nbsp;├── base.html  
&nbsp;&nbsp;&nbsp;├── index.html  
&nbsp;&nbsp;&nbsp;├── script.html  
&nbsp;&nbsp;&nbsp;└── search_results.html  

