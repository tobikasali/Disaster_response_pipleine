import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,precision_score


def load_data(database_filepath):
   '''
    This function loads the data from the database and output the features and target
    Input:
        database_filepath: OS location for sqlite database
    Output:
        X: Message 
        y: Categories
        colnames: Category Labels
        
    ''' 
   engine = create_engine('sqlite:///' + database_filepath)
   df = pd.read_sql_query("select * from messages ", engine)

   X = df.message.values
   y = df.iloc[:,4:40].values
   colnames = df.iloc[:,4:40].columns
    
   return X,y, colnames

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
        
    # lemmatize and remove stop words
    
    #stop_words = stopwords.words("english")
    #tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    Build the ML pipeline 
    Input: 
    Output:
        GridSearchCV pipleline
    '''


    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
         
    ])
    
    parameters =     {'clf__estimator__n_estimators': [50,100],
                      'clf__estimator__criterion': ['entropy', 'gini']
                 }
    
    model_cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return model_cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate the ML model.
    Input: 
        model: Machine learning model 
        X_test: test features( Messages)
        Y_test: test target lables (Categories)
        category_names: Category Labels
        
    Output:
        print  the f1 score, precision and recall for each category
    '''
    
    y_pred = model.predict(X_test)
    
    for idx,val in enumerate(category_names):
        print("Category:", val,"\n", classification_report(Y_test[:, idx], y_pred[:,             idx]))
        
    


def save_model(model, model_filepath):
    '''
    Input: 
        model: The trained model to be saved to disk.
        model_filepath: model store location on Disk
    '''
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


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