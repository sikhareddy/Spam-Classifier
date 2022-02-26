from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# load the model from disk
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('transform.pkl', 'rb'))
app = Flask(__name__, template_folder = "templates")

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
        
		message = request.form['message']
		data = [message]
		vect = cv.transform(data)
		prediction = clf.predict(vect)
	return render_template('result.html',prediction = prediction)



if __name__ == '__main__':
	app.run(debug=True)

