from nltk.sentiment.vader import SentimentIntensityAnalyzer
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def analyze_sentiment(comments):
    sid = SentimentIntensityAnalyzer()
    processed_comments = []
    for comment in comments:
        scores = sid.polarity_scores(comment)
        if scores['compound'] > 0:
            sentiment = 'Positive'
        elif scores['compound'] < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        processed_comments.append((comment, sentiment))
    return processed_comments

# ... (Load or create 'comments' and 'sentiments')

# Analyze comments using VADER
processed_comments = analyze_sentiment(comments)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processed_comments, sentiments, test_size=0.2, random_state=42)

# Vectorize comments
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform([comment for comment, _ in X_train])  # Extract just the comments
X_test_vec = vectorizer.transform([comment for comment, _ in X_test])

# Train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)

# Predict sentiments on the test set
y_pred = nb_classifier.predict(X_test_vec)

def sentiment_analysis_view(request):
    comments = [
        "Martín is a chad 😎"
        "Genuinely the best taught module this year, difficulty aside",
        "The content was interesting but the assessments were challenging.",
        "I wish they’d taken more time to talk about Agda, and the differences between it and Haskell. There are a lot of tricks that we were only taught after the first practice test that we didn’t know."
        "I don't particularly agree with this, I found that all the tricks I needed were already given before the practise test and that I could do well because of that, points were reiterated after the practise but it wasn't anything new beyond more details on constructors",
        "Not completely horrible if you were comfortable with Functional in 2nd year, and works quite well with the first half of PLPDI",
        "Emacs is a pain to learn, coming from someone using VSCode for everything."
        "The VSCode Agda extension is pretty bad, it didn’t let me use brackets, and half of the commands were broken."
        "I’d recommend installing Agda on your own machine, emacs via ssh is much worse than emacs on its own."
        # Add more comments as needed
    ]
    sentiments = analyze_sentiment(comments)
    return render(request, 'sentiment_analysis.html', {'sentiments': sentiments})


# Prepare your labeled dataset with comments and corresponding sentiments
# Load your dataset, or create it manually
comments = [...]  # List of comments
sentiments = [...]  # List of corresponding sentiments

# Perform sentiment analysis using VADER on the comments
sid = SentimentIntensityAnalyzer()
vader_sentiments = []
for comment in comments:
    scores = sid.polarity_scores(comment)
    if scores['compound'] > 0:
        sentiment = 'Positive'
    elif scores['compound'] < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    vader_sentiments.append(sentiment)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(vader_sentiments, sentiments, test_size=0.2, random_state=42)

# Vectorize text data (if needed)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)

# Predict sentiments on the test set
y_pred = nb_classifier.predict(X_test_vec)

# Evaluate classifier performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)