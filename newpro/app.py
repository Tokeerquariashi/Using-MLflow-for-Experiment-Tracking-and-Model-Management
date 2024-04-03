from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        review = request.form['text']
        model = joblib.load("nav.sav")
        prediction = model.predict([review])
        if prediction==0:
            ans="Negative"
        else:
            ans="Positive"
        return render_template('result.html', ans=ans)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)