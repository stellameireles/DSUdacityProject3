# DSUdacityProject3
Project 3 of the Data Scientist course at Udacity.

The main goal of the project is to build a model to analyse the messages of the dataset provided by Figure Eight and classify these messages.

# Libraries 

 NumPy
 Pandas
 nltk
 scikit-learn
 sqlalchemy


# Data 

disaster_categories.csv: message categories
disaster_messages.csv: multilingual disaster response messages

# Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
 http://view6914b2f4-3001.udacity-student-workspaces.com
