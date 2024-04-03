import pandas as pd
import mlflow
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
lemmatizer=WordNetLemmatizer()
df=pd.read_csv("data.csv")
mlflow.set_experiment("sentiment analysis")
df.dropna(inplace=True)
l=[]
for i in range(len(df['Ratings'])):
    if int(df['Ratings'].iloc[i]) >= 4:
        l.append('positive')
    else:
        l.append('negative')
df['re']=l
la=LabelEncoder()
val=la.fit_transform(df['re'])
import re
def preprocess_text(text):
    text = str(text)
    sentence = re.sub("[^a-zA-Z]", " ", text)
    sentence = sentence.lower()
    tokens = sentence.split()
    tokens = [t for t in tokens if not t in stopwords.words("english")]
    tokens = [lemmatizer.lemmatize(word,pos='v') for word in tokens]
    return pd.Series([" ".join(tokens), len(tokens)])
demo=df['Review text'].apply(preprocess_text)
df['review']=demo[0]
X_train,x_test,y_train,y_test=train_test_split(df['review'],val,test_size=0.3,random_state=0,stratify=val)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB


# Define pipelinefrom sklearn.naive_bayes import MultinomialNB
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
parameters = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],  
    'clf__alpha': [0.1, 1, 10]  
}

grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=1)
mlflow.sklearn.autolog(max_tuning_runs=None)

with mlflow.start_run() as run:
    grid_search.fit(X_train, y_train)

pipelines = {    
    'naive' : Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ]), 
    'logistic': Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ]),
    'random': Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier())
    ]),
    'decision_tree': Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', DecisionTreeClassifier())
    ])
}

param_grids={
    'naive':[
    {
        'clf': [MultinomialNB()],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__alpha': [0.1, 1.0, 10.0]
    }],
    'logistic':[
    {
        'clf': [LogisticRegression()],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.1, 1.0, 10.0],
        'clf__solver': ['liblinear', 'lbfgs']
    }],
    'random':[
    {
        'clf': [RandomForestClassifier()],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 10, 20]
    }],
    'decision_tree':[
    {
        'clf': [DecisionTreeClassifier()],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__max_depth': [None, 10, 20]
    }]
}
import time
import os
# Define parameter grid for each algorithm
dev = "Sivasankar"
best_models = {}

for algo in pipelines.keys():
    print("*"*10, algo, "*"*10)
    grid_search = GridSearchCV(estimator=pipelines[algo], 
                               param_grid=param_grids[algo], 
                               cv=5, 
                               scoring='accuracy', 
                               return_train_score=True,
                               verbose=1
                              )

    # Fit
    # start_fit_time = time.time()
    grid_search.fit(X_train, y_train)
    # end_fit_time = time.time()

    # Predict
    # start_predict_time = time.time()
    y_pred = grid_search.predict(x_test)
    # end_predict_time = time.time()

    # Saving the best model
    joblib.dump(grid_search.best_estimator_, f'best_models/{algo}.pkl')
    model_size = os.path.getsize(f'best_models/{algo}.pkl')

    # Pring Log
    print('Train Score: ', grid_search.best_score_)
    print('Test Score: ', grid_search.score(x_test, y_test))
    # print("Fit Time: ", end_fit_time - start_fit_time)
    # print("Predict Time: ", end_predict_time - start_predict_time)
    print("Model Size: ", model_size)
    
    print()

    # Start the experiment run
    # with mlflow.start_run() as run:
    #     # Log tags with mlflow.set_tag()
    #     mlflow.set_tag("developer", dev)

    #     # Log Parameters with mlflow.log_param()
    #     mlflow.log_param("algorithm", algo)
    #     mlflow.log_param("hyperparameter_grid", param_grids[algo])
    #     mlflow.log_param("best_hyperparameter", grid_search.best_params_)

    #     # Log Metrics with mlflow.log_metric()
    #     mlflow.log_metric("train_score", grid_search.best_score_)
    #     mlflow.log_metric("test_score", grid_search.score(X_test, y_test))
    #     mlflow.log_metric("fit_time", end_fit_time - start_fit_time)
    #     mlflow.log_metric("predict_time", end_predict_time - start_predict_time)
    #     mlflow.log_metric("model_size", model_size)

    #     # Log Model using mlflow.sklearn.log_model()
    #     mlflow.sklearn.log_model(grid_search.best_estimator_, f"{algo}_model")