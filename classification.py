# imports
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.model_selection import train_test_split, cross_val_score
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

#nltk.download('punkt')
#nltk.download('wordnet')

warnings.filterwarnings("ignore")

df = pd.read_csv('data\dataset - MANUAL FINAL.csv') 
df['Difficulty'] = df['Difficulty'].str.lower()
print(df['Difficulty'])
# Ensure dataset is balanced
print(df['Difficulty'].value_counts())

# Vectorize 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Question'])
y = df["Difficulty"]


k = 5
selector_mi = SelectKBest(mutual_info_classif, k=k)
X_new_mi = selector_mi.fit_transform(X, y)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_new_mi, y, test_size=0.2, random_state=42)

# Classifiers
nb = MultinomialNB()
svm = SVC(probability=True, kernel='linear')  # linear kernel for text data
knn = KNeighborsClassifier(n_neighbors=3)  

# Voting Classifier
ensemble_clf = VotingClassifier(
    estimators=[('nb', nb), ('svm', svm), ('knn', knn)], 
    voting='soft'
)

# Train the ensemble classifier
ensemble_clf.fit(X_train, y_train)

# Evaluate model using cross-validation
cv_scores = cross_val_score(ensemble_clf, X_train, y_train, cv=5)
print(f'\n\nCross-Validation Accuracy Scores: {cv_scores}')
print(f'\nAverage CV Accuracy: {np.mean(cv_scores)}')

# functions - WordNet similarity and semantic based classification 

def wordnet_similarity(question, target_word):
    question_words = word_tokenize(question)
    max_similarity = 0
    for word in question_words:
        synsets1 = wn.synsets(word)
        synsets2 = wn.synsets(target_word)
        for syn1 in synsets1:
            for syn2 in synsets2:
                similarity = syn1.wup_similarity(syn2)
                if similarity and similarity > max_similarity:
                    max_similarity = similarity
    return max_similarity

def classify_question_with_semantics(question):
    X_new = vectorizer.transform([question])
    X_new_selected = selector_mi.transform(X_new)  # Use the same feature selection method MI
    pred = ensemble_clf.predict(X_new_selected)[0]
    similarity = wordnet_similarity(question, pred.lower())
    return pred, similarity

# testing the model with example question

# Example classification
new_question = "Explain the concept of multipath propagation in wireless communication, discussing its impact on signal reception and diversity techniques."
classification, similarity = classify_question_with_semantics(new_question)

print(f"\n\nQuestion: {new_question}")
print(f"\n\nClassification: {classification}")
print(f"\nWordNet Similarity with '{classification.lower()}': {similarity*100} %")

