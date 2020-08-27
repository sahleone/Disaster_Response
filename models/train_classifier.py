import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report

from sklearn.base import BaseEstimator, TransformerMixin
import pickle




def avg_lenght_(X_):
    avg_lenght = list(map(lambda x:len(x.replace(' ',''))/len(x.split(' ')), X_))
    return np.array(avg_lenght).reshape(-1,1)

def lenght_(X_):
    lenght= list(map(lambda x : len(x.split(' ')), X_))
    return np.array(lenght).reshape(-1,1)

class avg_lenght(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X_, y=None):
        return self

    def transform(self, X_):
        return avg_lenght_(X_)
    
class lenght(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X_, y=None):
        return self

    def transform(self, X_):
        return lenght_(X_)





def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster_Response_Message', engine)

    X = df['message']
    Y = df.drop(['id', 'message', 'genre'], axis = 1)
    category_names = list(Y.columns)
    return X,Y,category_names

def tokenize(text):
    
    '''Cleans Text Data by:
    1.Remove stopwords
    2.Remove punctuation
    3.Normalize text
    4.Lemmatize text'''
    
    import re
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer

    stop_words =  set(stopwords.words('english'))
    
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize(noun) and remove stop words
    lem = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # lemmatize(verb) and remove stop words
    lem = [lemmatizer.lemmatize(word,pos = 'v') for word in lem ]
    # lemmatize(adjective) and remove stop words
    tokens  = [lemmatizer.lemmatize(word,pos = 'a') for word in lem ]

    return tokens


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize,ngram_range =(1,2))),
                ('tfidf', TfidfTransformer())
            ])),

            ('avg_lenght', avg_lenght()),
            ('lenght',lenght())
        ])),
    
    
        ('clf', MultiOutputClassifier(RandomForestClassifier(\
        class_weight = 'balanced',max_depth= 8,n_estimators= 250)))
        ])
    
    return pipeline




def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_test_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_test_pred, target_names = category_names))


def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()