from flask import Flask, render_template, url_for, request, jsonify      
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
import numpy as np
from sklearn.linear_model import Ridge

app = Flask(__name__)

# Load the TF-IDF vocabulary specific to the category
with open(r"upvote_vect.pkl", "rb") as f:
    upvote = pickle.load(f)

# Load the pickled RDF models
with open(r"upvote_model.pkl", "rb") as f:
    upvote_model = pickle.load(f)

# Render the HTML file for the home page
@app.route("/")
def home():
    return render_template('index_toxic.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    # Take a string input from user
    user_input = request.form['text']
    data = [user_input]

    vect = upvote.transform(data)
    pred = upvote_model.predict(vect)
    

    
    # out_upvote = round(pred_upvote[0], 0)
    

    # print(out_upvote)

    return render_template('index_toxic.html', 
                            pred_upvote =pred                   
                            )

     
# Server reloads itself if code changes so no need to keep restarting:
app.run(debug=True)
app.run(host='0.0.0.0', port=8080)