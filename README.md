# Disaster_response_pipleine


### Table of Contents

1. Project Summary & Motivation
2. Contents of the Respository 
3. Running the scripts and web app
4. Summary of results
5. Licensing and Acknowledgements

## Project Motivation

The purpose of this project is to analyse a disaster data set from figure Eight to build for a model that classifies disaster messages. The project will also include a web ap which will display visulaisations of the data and also show the results of the data model. The project will involve taking the messages as input and classifying them into one of 36 categories of help that could be required. This will help Aid organisation plan better and spend less time sorting out messages .


## Contents of the Repository

Readme.md:  This file, describing the contents of this repo.

app: The folder containning the files to run the application.
 * templates:
 * run.py: The python script to run the wep app.
 
data: The folder containing the dataset, process data script and storesthe output of the process data script:database. 
 * DisasterResponse.db - The csv file containing the Seattle AirBnB listings.
 * disaster_categories.csv - The csv file containing the disaster categories.
 * disaster_messages.csv - The csv file containing the disaster messages.
 * process_data.py - Script that cleans and stores the data in a database.

models:The folder containing the scripts of the model and the saved model.
 * classifer.pkl: The saved model and output of the  model.
 * train_classify.py: The python script to train the model and build a machine learning model.

## Running the scripts and webapp

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/.

### Summary of results

The model shows F1, precision and recalls score for each of the categories of data.

### Licenses and Acknowledgements

The dataset belongs to figure 8 with other licnesing information. 
