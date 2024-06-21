# A demo code of NLP Pre-processing on Movie reviews Dataset adn understanding Count Vectorizer......................
# Here we use NLTK for cleaning and create the data into format for SKLearn classifier

from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import movie_reviews
from nltk.corpus import wordnet
import random
from sklearn.feature_extraction.text import CountVectorizer


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


def cleaning_file(document):
    cleaned_words=[]
    for i in range(len(document)):
        new_array=cleaning_words(document[i][0])
        # print(new_array)
        cleaned_words.append((new_array,document[i][1]))
    return cleaned_words

# Creating Sentences for Count Vectorizer to find top features...
def creating_sentences(x_y_train):
    x_train_sentences=[]
    y_categories=[]
    for i in range(len(x_y_train)):
        x_train_sentences.append(" ".join(x_y_train[i][0]))
        y_categories.append(x_y_train[i][1])
    return x_train_sentences,y_categories

# Performing Count Vectorization.......
def count_vectorize(x_train_sentences,x_test_sentences):
    count_vec=CountVectorizer(max_features=2000)
    # Count vectorizer has also other features/parameters that could be used to increase accuracy in some cases, such as:
    # 1. n-gram:-It will create tokens of 2 words also, compared to earlier where we used only single words as features.
    #            Single word as feature- uni-gram, Two words as feature:- Bi-gram

    # 2. TF-IDF: Term Frequency or Inverse Document Frequency.
    #            IDF(w)= 1/D.F(w)   ;   D.F:- No. of documents present that contain the word w
    #            It is done so that if a word that is used in every document then it doesn't provide much information. IDF is used to eliminate it.
    #            T.F(w):- It is the no of words present in the document.
    # The count vectorizer only utilizes the term frequency and doesn't use the IDF.
    # To use IDF, either multiply each word with its IDF value after the count vectorizer has calculated the sparse matrix or use the TF-IDF Vectorizer.

    # While Building the vocabulary, we use max_df and min_df parameters to ensure that words with very high frequency and words with very low frequency are eliminated.
    # e.g max_df=0.9 ensures that word that occurs in more than 90% documents be removed from feature set.
    # Similarly min_df=0.1 ensures that word that occur in less than 10% documents be eliminated.

    # Use N-gram/feature in this function, like pass this paramterer: ngram=(1,3) for uni,bi and tri grams.
    # Use the max_df or min_df feature to eliminate words which don't provide much information.
    x_train_features=count_vec.fit_transform(x_train_sentences)

    print(count_vec.get_feature_names())
    # This feature is important as we get the sparse matrix which could be used by the sklearn classifiers.
    print(x_train_features.todense())
    x_test_features=count_vec.transform(x_test_sentences)
    return x_train_features,x_test_features

# Performing all the loading and cleaning on dataset..............
document=loading()
# print(document)
cleaned_words=cleaning_file(document)
x_y_train, x_y_test=cleaned_words[0:1500],cleaned_words[1500:]

# Using Count Vectorizer..........
x_train_sentences,y_train_categories=creating_sentences(x_y_train)
x_test_sentences,y_test_categories=creating_sentences(x_y_test)


# Count Vectorizer takes all the training data, that is the text after being cleaned and find the mentioned number of top features.
# Then it creates a sparse matrix which will hold values for all words where if the word is in document an if it's selected as a feature it would
# be one and matrix would hold it's frequency value and if the word is in not in document and present in feature it would be zero. If the word is not
# e.g:- Count Vectorizer:

train_demo={'the sky is blue','the sun is bright'}
count=CountVectorizer(max_features=3)
print(count.fit_transform(train_demo))
print(count.fit_transform(train_demo).todense())






# This returns a sparse matrix for sklearn classifier.
# Note: Count Vectorizer fit_transform has to be run only on training set and not testing data.
# On Testing data count_vectorizer_object.transform() needs to be called to transform data according to the learned parameters.
x_train_features,x_test_features=count_vectorize(x_train_sentences,x_test_sentences)

# Using the Random Forest Classifier..................
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train_features,y_train_categories)
y_pred=rf.predict(x_test_features)

# Analysing the Output.........
from  sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test_categories,y_pred))
print(confusion_matrix(y_test_categories,y_pred))

