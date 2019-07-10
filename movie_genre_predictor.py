import re
import csv
import nltk
import json
import argparse
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


# creates a genre predictor/classifier which does multi-class (many genres) and multi-label (each movie may have more than one genre) classification
class MovieGenrePredictor():
    # init function of the class where the classifier training occurs
    def __init__(self):
        # read the movies_metadata.csv dataset from https://www.kaggle.com/rounakbanik/the-movies-dataset/#movies_metadata.csv into a dataframe 
        df = pd.read_csv('the-movies-dataset/movies_metadata.csv',sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
        
        # remove other coulumns as we only need overview and genres for our genre prediction task. I have kept 'id' column only for help during debugging
        df.drop(df.columns.difference(['overview','id','genres']), 1, inplace=True)

        # extract list of genres from the column containing genres in json format
        genres = [] 
        for i in df['genres']:
            g = json.loads(i.replace("\'", "\""))
            df_g = pd.DataFrame(g)
            values = df_g.get('name')
            l = []
            if values is not None:
                l = values.tolist()
            genres.append(l)

        # add these list of genres to the dataframe
        df['genres_new'] = genres

        # remove any rows that didn't have any genres, as they will be useless to build our model
        df = df[~(df['genres_new'].str.len() == 0)]

        # get all genres in a list and create a frequency distribution of them to evaluate if we have any illogical genres in the data
        all_genres = sum(genres,[])
        all_genres = nltk.FreqDist(all_genres)
        all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 
                                      'Count': list(all_genres.values())})
        
        # remove genres which like 'Carousel Productions' and 'The Cartel' which have count less than 10 i.e. they are probably mistakes in the data
        all_genres_df = all_genres_df[all_genres_df['Count'] > 10]
        all_genres = all_genres_df['Genre'].tolist()

        # remove data entries having the genres not in list of (valid) genres
        invalid_df = df[df['genres_new'].apply(lambda x: any([k in x for k in all_genres]))==False]
        df = df.drop(invalid_df.index.values)

        # remove data rows where overview is NaN or empty
        df = df.dropna(axis=0, subset=['overview'])
        df = df[~(df['overview'].str.len() == 0)]

        # preprocess the overview column by removing punctuations, numbers, and lowercasing all text, as well as removing stopwords like 'the','is', etc
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        df['overview_clean'] = df['overview'].apply(lambda x: self.clean_text(str(x)))

        # encoding the 'genres_new' values into a mask of binary target values, each indicating the presence(1) or absense(0) of a genre
        self.multilabel_binarizer = MultiLabelBinarizer()
        self.multilabel_binarizer.fit(df['genres_new'])
        y = self.multilabel_binarizer.transform(df['genres_new'])

        # split dataset into training and validation set
        xtrain, xval, ytrain, yval = train_test_split(df['overview_clean'], y, test_size=0.2, random_state=11)

        # create TF-IDF features from training and validation sets
        self.tfidf_vectorizer = TfidfVectorizer()
        xtrain_tfidf = self.tfidf_vectorizer.fit_transform(xtrain)
        xval_tfidf = self.tfidf_vectorizer.transform(xval)

        # use a OneVsRest classifier using Linear Support Vector classification
        self.classifier = OneVsRestClassifier(LinearSVC())

        # fit model on the training set
        self.classifier.fit(xtrain_tfidf, ytrain)

        # make predictions for validation set
        y_pred = self.classifier.predict(xval_tfidf)

        # evaluate performance (returns around 0.52 currently)
        # f1 = f1_score(yval, y_pred, average="micro")
        # print(f1)

     # function for cleaning descriptive text
    def clean_text(self, text):
        # remove backslash-apostrophe 
        text = re.sub("\'", "", text) 
        # remove everything except alphabets 
        text = re.sub("[^a-zA-Z]"," ",text) 
        # remove whitespaces 
        text = ' '.join(text.split()) 
        # convert text to lowercase 
        text = text.lower()
        # remove stopwords from the text
        no_stopword_text = [w for w in text.split() if not w in self.stop_words]
        return ' '.join(no_stopword_text)

# the output of the program is the json version of this class
class Output():
    def __init__(self,title,description,genre):
        self.title = title
        self.description = description
        self.genre = genre


# create an argument parser to input the title and description from command line in the required format
parser=argparse.ArgumentParser()
parser.add_argument('--title', help='the movie title', type= str)
parser.add_argument('--description', help='the movie description', type= str)
args=parser.parse_args()

# create the predictor using the arguments passed
mgp = MovieGenrePredictor()
description = args.description
description = mgp.clean_text(str(description))
description_tfidf = mgp.tfidf_vectorizer.transform(pd.Series(description))
output_genre_vector = mgp.classifier.predict(description_tfidf)
output_genre = mgp.multilabel_binarizer.inverse_transform(output_genre_vector)

# create the object of Output class which is then dumped as json and printed as output.
output = Output(args.title,args.description,output_genre)
j = json.dumps(output.__dict__, indent = 4)
print(j)