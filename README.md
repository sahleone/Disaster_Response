
# Data Science - Disaster Response Message Classification
This project is part of the Data Science course offered by Udacity. The goal of the project is to analyze disaster data and build a model that can classify disaster messages. The project consists of three main components: ETL Pipeline, ML Pipeline, and Flask Web App.

## Project Overview
In this project, data engineering skills to analyze disaster data obtained from Figure Eight. The objective is to build a model for an API that can classify disaster messages. The project includes the development of a web app where an emergency worker can input a new message and receive classification results in multiple categories. The web app also provides data visualizations.

## Screenshots of the Web app
![](Disaster_Response_Screenshot1.png)
![](Disaster_Response_Screenshot2.png)

## Project Components
The project is divided into three components:

1. ETL Pipeline: A Python script, process_data.py, that performs the Extract, Transform, and Load process. It loads and merges the messages and categories datasets, cleans the data, and stores it in a SQLite database.

2. ML Pipeline: A Python script, train_classifier.py, that builds a machine learning pipeline. It loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline using NLTK and scikit-learn, trains and tunes a model using GridSearchCV, and exports the final model as a pickle file.

3. Flask Web App: A web app that allows an emergency worker to input a message and receive classification results. It also displays data visualizations based on the data extracted from the SQLite database.
## Installation
In the project's root directory, run the following commands:

### ETL pipeline
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
### ML pipeline
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

### Render the Web App
1. In the app's directory run:
    `python run.py`

2. Go to http://0.0.0.0:3001/

## Acknowledgments
The project was done to satisfy the Data Science Nanodegree. It uses data from Figure Eight.

## Licence

MIT License

Copyright (c) [2020] [Rhys Jervis]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


