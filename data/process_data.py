import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Input:
        messages_filepath: File location of messages data
        categories_filepath: File location of categories data
    Output:
        df: merged dataset of messages and categories
     '''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    #messages.head()
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    #categories.head()
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
       
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    row = row.str.split('-').str[0]
    
    # use this row to extract a list of new column names for categories.

    category_colnames = row
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]
        val = categories[column]
    
    # convert column from string to numeric
        categories[column] = pd.to_numeric(val)
        categories[column] = categories[column].astype(int)
            
            
    # Replace categories column in df with new category columns.      
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1,)
    
    ## Drop any rows that have Nan values in the category columns. 
    ## The message column has no Nan's
    df= df.dropna(subset = [category_colnames])
    
    ## change the category values to int
    for cat in category_colnames:
         df[cat] = df[cat].astype(int) 
    
    return df


def clean_data(df):
    '''
    Input:
        df: dataset of messages and categories
    Output:
        df: cleaned dataset of messages and categories
     '''
    
    # drop duplicates
    df = df.drop_duplicates(keep="last")
    
    return df


def save_data(df, database_filename):
    '''
    Input:
        df: dataset of messages and categories
        database_filename: "name of the databasefile.db"
    Output:
        df: cleaned dataset of messages and categories
     '''
    
    engine = create_engine('sqlite:///' +database_filename)
    df.to_sql('messages', engine, if_exists='replace', index=False)
     


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()