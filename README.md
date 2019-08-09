# movie-genre-predictor
Predicts genres of a movie, given a description.
Uses MovieLens dataset (https://www.kaggle.com/rounakbanik/the-movies-dataset/#movies_metadata.csv) and creates a multi-label classifier.

My Setup:
- Python Version 3.7.3

External Python libraries needed to be preinstalled:
- 'nltk' for getting frequency distribution of words as well as getting common stopwords
- 'pandas' for working with data frames
- 'sklearn' for using classifiers, train_test_split, f1_score

Steps for running:
- Download/clone the repository (this includes the data in the-movies-dataset folder)
- Using your terminal, navigate to contents of the repository and execute command in the following syntax:
- python3 movie_genre_predictor.py --title "title" --description "description"

For example:  
python3 movie_genre_predictor.py --title "Avengers: Endgame" --description "After the devastating events of Avengers: Infinity War (2018), the universe is in ruins. With the help of remaining allies, the Avengers assemble once more in order to reverse Thanos' actions and restore balance to the universe."

The output is in the following json format, with the predicted genre(s) at the bottom:  
{  
    "title": "Avengers: Endgame",  
    "description": "After the devastating events of Avengers: Infinity War (2018), the universe is in ruins. With the help of remaining allies, the Avengers assemble once more in order to reverse Thanos' actions and restore balance to the universe.",  
    "genre": [  
        "Action",  
        "Science Fiction"
    ]  
}  
