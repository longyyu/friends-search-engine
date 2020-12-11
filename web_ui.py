from flask import Flask, render_template, redirect, url_for, request, abort
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import SelectField, StringField, SubmitField
from wtforms.validators import DataRequired
from datetime import datetime

import os
#from metapy import metapy
#from config_metapy import config_file, inv_idx, get_retrieval_results
from data_prep import script_utterance, get_script_with_uid
from inverted_index import indexes, get_retrieval_results

# Purpose: This script #! add header
# Author: Yanyu Long
# Updated: Dec 11, 2020

app = Flask(__name__)
Bootstrap(app)


class SearchForm(FlaskForm):
  character = SelectField("Filter by character", 
    choices = [("", "Select a character"), 
                ("Chandler Bing", "Chandler"), 
                ("Joey Tribbiani", "Joey"), 
                ("Monica Geller", "Monica"),
                ("Phoebe Buffay", "Phoebe"), 
                ("Rachel Green", "Rachel"), 
                ("Ross Geller", "Ross")]
  )
  user_query = StringField(
    validators = [DataRequired()],
    render_kw = {
      "placeholder": "Joey doesn't share food!",
      "style": "width: 500px; font-size: 15px;"
    }
  )
  search_button = SubmitField("Search!")


@app.route("/", methods=["GET", "POST"])
def index():
  sample_search_img = [item.replace(".png", "") 
    for item in os.listdir("./static/sample-search/")
  ]
  search_form = SearchForm(meta={'csrf': False})
  if search_form.validate_on_submit():
    return redirect(url_for("search_results", 
      query = search_form.user_query.data,
      character = search_form.character.data
    ))
  return render_template("index.html", 
                         form = search_form,
                         imgs = sample_search_img)


@app.route("/search_results/q=<query>", 
           methods=["GET", "POST"], defaults={'character': ""})
@app.route("/search_results/q=<query>/c=<character>", methods=["GET", "POST"])
def search_results(query, character):
  start_time = datetime.now()
  # result_list = get_retrieval_results(
  #   query_content = query,
  #   ranker =  metapy.index.OkapiBM25(k1 = 1.2, b = 0.75, k3 = 500),
  #   inv_idx = inv_idx,
  #   df_uid = script_utterance,
  #   num_results = 10 #! should not limit the # of results?
  # )
  result_list = get_retrieval_results(
    query = query, filter_by_character = character
  )
  docs = [get_script_with_uid(
            df = script_utterance, 
            u_id = u_id, 
            plus_minus = 1,
            output_format = "html"
          ) for u_id in result_list]
  finish_time = datetime.now()

  search_form = SearchForm(meta={'csrf': False})
  if search_form.validate_on_submit():
    return redirect(url_for("search_results", 
      query = search_form.user_query.data,
      character = search_form.character.data
    ))
  
  return render_template("search_results.html",
                         processing_time = (finish_time - start_time),
                         total_doc_num = len(docs),
                         query = query,
                         character = character,
                         docs = docs,
                         form = search_form)

if __name__ == "__main__":
  app.run(threaded = False)
