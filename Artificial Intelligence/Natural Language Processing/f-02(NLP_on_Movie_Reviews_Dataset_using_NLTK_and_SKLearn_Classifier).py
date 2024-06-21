# A demo code of NLP Pre-processing on Movie reviews Dataset using NLTK classifier......................

from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import movie_reviews
import random
from nltk.corpus import wordnet
import nltk

from collections import Counter
# Counter(a).most_common(1)[0][0]

# The Loading Function takes the document/collection of words and returns a dataset of that is a list of tuples that has collection of words and its corresponding class.
def loading():
    # print(movie_reviews.categories())
    # print(movie_reviews.fileids())
    documents=[]
    for i in movie_reviews.categories():
        for j in movie_reviews.fileids():
            documents.append((movie_reviews.words(j),i))
    # print(documents[0:5])
    random.shuffle(documents)
    return documents

# Returns the Corresponding characteristic that is noun/adjective etc. in a way that can be used for lemmatization.
def pos_to_wordnet(pos_tag):
    # print(pos_tag)
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('R'):
        return wordnet.ADJ
    else:
        return wordnet.NOUN

# The function cleaning_words removes the stopwords and punctuations and returns the cleaned words array.
# Then this function lemmatizes each word and gets the root word.
def cleaning_words(words):
    word_array=[]
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    punctuations = list(string.punctuation)
    stop_words += punctuations
    for i in range(len(words)):
        if words[i] in stop_words:
            continue
        else:
            pos_tuple_returned=pos_tag([words[i]])
            word=lemmatizer.lemmatize(words[i],pos=pos_to_wordnet(pos_tuple_returned[0][1]))
            word_array.append(word)
    return word_array

# Loops over the each document and returns the cleaned dataset
def cleaning_file(document):
    cleaned_words=[]
    for i in range(len(document)):
        new_array=cleaning_words(document[i][0])
        # print(new_array)
        cleaned_words.append((new_array,document[i][1]))
    return cleaned_words

# From all the documents, we select the top words for features of the dataset
def finding_features(document):
    feature_words=[]

    for i in range(len(document)):
        feature_words+=(document[i][0])
    print(type(feature_words))
    # Frequency Distribution Object.................
    freq= nltk.FreqDist(feature_words)
    top_words_tuple=freq.most_common(3000)
    top_words=[i[0] for i in top_words_tuple]
    # print(top_words)
    return top_words

def creating_dictionary_single_file(top_words,words):
    dict={}
    for i in range(len(words)):
        if words[i] in top_words:
            dict.update({words[i]: True})
        else:
            continue
    return dict


# Create the format which is required by NLTK classifier.
# For the NLTK Classifier we have to have a specific format.
#   list[tuples:-(dict:- {document_1:- {word_1: freq(word_1),
#            word_2: freq(word_2),
#            .......
#            word_n: freq(word_n)}})]


def creating_dictionary(top_words,document):
    feature_set=[]
    for i in range(len(document)):
        dict=creating_dictionary_single_file(top_words,document[i][0])
        feature_set.append((dict,document[i][1]))
    return feature_set


# Importing multiple Classifiers and NLTK Classifier Handlers
from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk import NaiveBayesClassifier

document=loading()
# print(document)
cleaned_words=cleaning_file(document)
x_y_train, x_y_test=cleaned_words[0:1500],cleaned_words[1500:]
top_words=finding_features(x_y_train)
feature_set=creating_dictionary(top_words,x_y_train)
test_set=creating_dictionary(top_words,x_y_test)


# Training cleaned data on using NLTK based Naive Bayes Classifier...................
classifier=NaiveBayesClassifier.train(feature_set)
print('Naive Bayes Classifier Accuracy:',nltk.classify.accuracy(classifier,test_set))
print(classifier.show_most_informative_features(15))


# Training cleaned data on Multiple Classifiers using SKLearn Classifier..................................
# The Sklearn Classifier can be used with the created NLTK format by modelling that format to SKLearn type classifier using NLTK SKLearnClassifer object.
# Using SVM and Random Forest with NLP Cleaning Techniques and NLTK classification.
svm=SVC()
random_forest=RandomForestClassifier()

# This dummy classifier changes every data format accordingly.....
classifier_sklearn=SklearnClassifier(svm)
classifier_sklearn_1=SklearnClassifier(random_forest)

# Training Classifiers.........
classifier_sklearn.train(feature_set)
classifier_sklearn_1.train(feature_set)

print('SVM Classifier Accuracy:',nltk.classify.accuracy(classifier_sklearn,test_set))
print('Random Forest Classifier Accuracy:',nltk.classify.accuracy(classifier_sklearn_1,test_set))

