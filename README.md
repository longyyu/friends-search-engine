# Search Engine for Friends Quotes  

Term project for SI650 / EECS549 at the University of Michigan.  
A vertical search engine for scripts of FRIENDS, the popular American TV sitcom produced by Warner Bros.  
Visit the web app [here](http://longyyu.pythonanywhere.com/)!&nbsp;&nbsp;
<span style="color: Sienna; background-color: WhiteSmoke; font-style: italic;">
  link to web app last updated on Dec 15, 2020
</span>

## Installation  
<span style="color: Sienna; background-color: WhiteSmoke; font-style: italic;">
  Note: this project is written in Python3 (3.7). 
</span>

* clone this repository  
* install the required packages: `pip3 install -r requirements.txt`
* download NLTK's sentence tokenizer `Punkt` by running `import nltk; nltk.download('punkt')` in Python

## Usage  
* change directory to the project's root folder
* run `python -m web_ui`

## Source files  

<span style="color: Sienna; background-color: WhiteSmoke; font-style: italic;">
  &nbsp;&nbsp;&nbsp;major Python scripts  
</span>  

├── config_metapy.py *# set up the baseline model (metapy)*  
├── data_prep.py *# read in and pre-process data*  
├── helper_func.py *# defines helper functions*  
├── inverted_index.py *# defines class Indexes, which builds up inverted index and ranks documents*  
├── ranker_evaluation.py *# evaluates ranker performance using AP and NDCG*  
├── web_ui.py *# defines the flask framework of the web app*  
├── **data**  
<span style="color: Sienna; background-color: WhiteSmoke; font-style: italic;">
  the original FRIENDS corpus data
</span>  
│   ├── json.tar.gz *# original data in JSON format, archived*  
│   ├── script_id_speaker_10seasons.tsv *# pre-processed FRIENDS corpus data*  
<span style="color: Sienna; background-color: WhiteSmoke; font-style: italic;">
  data for building and evaluating retrieval models
</span>  
│   ├── stopwords.txt *# stop words data*  
│   ├── friends-qrels.txt *# query judgement data for ranker evaluation purpose*  
│   ├── friends-queries.txt *# testing queries*  
<span style="color: Sienna; background-color: WhiteSmoke; font-style: italic;">
  data/files for building the baseline model using package `metapy`
</span>  
│   ├── friends-config.toml *# metapy config file*  
│   ├── friends/... *# data input for metapy*  
│   ├── friends-idx/... *# inverted index built by metapy*  
<span style="color: Sienna; background-color: WhiteSmoke; font-style: italic;">
  temporary data saved locally to expedite application setup
</span>  
│   ├── term_to_freq_pos.pkl  
│   ├── corpus_term_freq.pkl  
│   └── doc_tokens.pkl  
<span style="color: Sienna; background-color: WhiteSmoke; font-style: italic;">
  files for the web app's user interface
</span>  
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

