#This is purely a test file to test if ensembling multiple classifiers together results in improved performance
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


def load_comments(filename=''):
    data = pd.read_csv(filename, header=None, names=['ID', 'Topic', 'Sentiment', 'Comment']) # read CSV file
    data = data.dropna(subset=['Comment']) # drop any rows with a sentiment but no comment/empty comment
    data = data[data['Sentiment'] != 'Irrelevant'] # drop any rows with the sentiment "Irrelevant" as we aren't using that for our predictions
    return data['Comment'], data['Sentiment']

def clean_comments(comments):
    stopwordslist = set(stopwords.words('english'))
    tokeniser = TweetTokenizer()
    lemmatiser = WordNetLemmatizer()
    clean_comments = []
    for comment in comments:
        # case for if the comment is NaN, just continue
        if pd.isna(comment):
            clean_comments.append([])
            continue
        words = tokeniser.tokenize(comment.lower()) # convert comment to lowercase and split it into words using tokeniser
        lemmatised_words = []
        for word in words:
            if word.isalpha(): # Check if word consists of only alphabetic characters
                if word not in stopwordslist:
                    lemmatised_word = lemmatiser.lemmatize(word) #lemmatise word assuming it's not in the list of stop words
                    lemmatised_words.append(lemmatised_word)
        clean_comments.append(lemmatised_words)
    return clean_comments

comments, sentiments = load_comments('twitter_training.csv') #training dataset
cleaned_comments = clean_comments(comments) # clean the comments in the dataset


pipeline = Pipeline([ # pipeline to ensemble multiple models together
    ('tfidf', TfidfVectorizer(stop_words='english')), # remove common english words (e.g "the") and transform text data into TF-IDF vectors
    ('clf', VotingClassifier([ # voting classifier to combine multiple classifiers to classify the data
        ('nb', MultinomialNB()), # naive bayes classifier
        ('svm', SVC(kernel='linear', probability=True)), # svm (support vector machines) classifier with linear kernel
        ('rf', RandomForestClassifier()), # random forest classifier
    ], voting='soft')) # soft voting takes the probabilities of each classifier into account
])

param_grid = { # dictionary to specify parameters that are gonna be used for grid search
    'tfidf__max_features': [1000, 2000, 3000], # test multiple values to determine optimal number of features to get best performance from model
    'clf__svm__C': [0.1, 1, 10], # avoid over/under fitting by testing different values for regularisation
    'clf__rf__n_estimators': [50, 100, 200], # different number of trees for the random forest classifier, more = better (but takes longer)
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1) # grid search to compare models based on accuracy, uses pipeline and specified parameters with 3-fold cross-validation, and uses all available CPU cores to make it faster
grid_search.fit(cleaned_comments, sentiments) # do the grid search on the cleaned comments, sentiments are target outputs
best_model = grid_search.best_estimator_ # find best model
cv_scores = cross_val_score(best_model, cleaned_comments, sentiments, cv=3, scoring='accuracy', n_jobs=-1, verbose=1) # 3-fold cross-validation on (cleaned) training data using the best model
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean()) # take mean to get an overall estimate of accuracy