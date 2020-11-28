from flask import Flask, render_template, redirect, url_for, request, abort
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from datetime import datetime

from metapy import metapy
from config_metapy import config_file, inv_idx, get_retrieval_results
from data_prep import script_utterance, get_script_with_u_id


app = Flask(__name__)
Bootstrap(app)


class SearchForm(FlaskForm):
    user_query = StringField(
      validators = [DataRequired()],
      render_kw = {
        "placeholder": "Joey doesn't share food!",
        "style": "width: 500px;"
      }
    )
    search_button = SubmitField("Search!")


@app.route("/", methods=["GET", "POST"])
def index():
    search_form = SearchForm(meta={'csrf': False})
    if search_form.validate_on_submit():
        return redirect(
          url_for("search_results", query = search_form.user_query.data)
        )
    return render_template("index.html", form = search_form)


@app.route("/search_results/<query>", methods=["GET", "POST"])
def search_results(query):
    start_time = datetime.now()
    result_list = get_retrieval_results(
      query_content = query,
      ranker =  metapy.index.OkapiBM25(k1 = 1.2, b = 0.75, k3 = 500),
      inv_idx = inv_idx,
      df_uid = script_utterance,
      num_results = 10 #! should not limit the # of results?
    )
    docs = [get_script_with_u_id(
              df = script_utterance, 
              u_id = u_id, 
              plus_minus = 1,
              output_format = "html"
            ) for u_id in result_list]
    finish_time = datetime.now()

    search_form = SearchForm(meta={'csrf': False})
    if search_form.validate_on_submit():
      return redirect(
        url_for("search_results", query = search_form.user_query.data)
      )
    
    return render_template("search_results.html",
                           processing_time = (finish_time - start_time),
                           total_doc_num = len(docs),
                           query = query,
                           docs = docs,
                           form = search_form)

if __name__ == "__main__":
    app.run(threaded = False)
