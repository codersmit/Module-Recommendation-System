import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(word): # improve accuracy of lemmatisation by determining part-of-speech tag from the context of a word in a sentence (e.g if word is tagged as verb, lemmatiser reduces word to its root form considering verb rules)
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def load_comments(filename):
    data = pd.read_csv(filename, header=None, names=['Comment', 'Commenter', 'Date', 'Rating', 'ID']) # read training data from CSV file
    data['Rating'] = data['Rating'].apply(lambda rating: int(rating) if str(rating).isdigit() else 0) # convert ratings from strings to integers, 0 if error (like AttributeError)
    # convert each comment's rating to positive, neutral, or negative in numerical form, then store it in a new column
    def convert_rating(rating):
        if rating > 3:
            return 1  # if rating is 4 or 5, return 1 for positive
        elif rating == 3:
            return 0.5  # return 0.5 for neutral
        else:
            return 0  # if rating is 1 or 2, return 0 for negative
    data['Converted_Rating'] = data['Rating'].apply(convert_rating)
    # randomly sample up to 50k comments from each group to ensure balanced dataset
    sample_size = 50000
    balanced_data = data.groupby('Converted_Rating').apply(lambda group: group.sample(n=min(len(group), sample_size), random_state=42)).reset_index(drop=True)
    return balanced_data['Comment'], balanced_data['Converted_Rating']

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
                    lemmatised_word = lemmatiser.lemmatize(word, get_wordnet_pos(word)) #lemmatise word assuming it's not in the list of stop words
                    lemmatised_words.append(lemmatised_word)
        clean_comments.append(lemmatised_words)
    return clean_comments

def vectorise_comments(comment, model):
    vectors = []
    for tokens in comment:
        vectors_for_tokens = []
        for token in tokens: # Loop through list of tokens
            if token in model.wv: # to check if token is in the model's vocabulary
                vectors_for_tokens.append(model.wv[token]) # If token was in the model's vocabulary, get its vector
        if not vectors_for_tokens:
            vectors.append(np.zeros(model.vector_size)) # Fix size error by appending zero vector of same length as vectors if no vectors were added (no tokens in vocabulary)
        else:
            mean_vector = np.mean(vectors_for_tokens, axis=0) # Calc the mean vector which will represent the whole comment
            vectors.append(mean_vector)
    return np.array(vectors)

def train_model(comments, ratings):
    comment = clean_comments(comments) # Clean the comments
    word2vec_model = Word2Vec(sentences=comment, vector_size=100, window=5, min_count=2, workers=4) # Model that turns words into vectors to learn the relationships between words from the comments
    vectors_for_comments = vectorise_comments(comment, word2vec_model) # Turn the words in each comment into vectors
    ratings_list = np.array(ratings)
    training_data, testing_data, training_labels, testing_labels = train_test_split(vectors_for_comments, ratings_list, test_size=0.2, random_state=42) # 80/20 split for training/testing sets
    rfg_model = RandomForestRegressor(n_estimators=100, random_state=42) # 100 trees for rfg model (for now)
    rfg_model.fit(training_data, training_labels) # Train the model
    # Save the trained model and word2vec model into pkl file to avoid having to train multiple times
    with open('random_forest_regressor.pkl', 'wb') as rfgmodel_file:
        pickle.dump(rfg_model, rfgmodel_file)
    with open('word2vec_model.pkl', 'wb') as f:
        pickle.dump(word2vec_model, f)
    return rfg_model, word2vec_model


if __name__ == "__main__":
    comments, ratings = load_comments('Coursera_reviews.csv') # Training dataset
    model, word2vec_model = train_model(comments, ratings) # Traininggggg