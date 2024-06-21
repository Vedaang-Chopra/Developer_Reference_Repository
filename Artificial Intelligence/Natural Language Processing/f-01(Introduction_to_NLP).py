# Using NLTK for all Natural Language Processing Techniques.
# nltk has to be downloaded on every first iteration when the code is run.

# import nltk
# nltk.download ()
# Note: Use this for Colab: !python -m nltk.downloader all
# Sample Text.....................
sample_text_1="Joey says, 'How you doing!!'"
sample_text_2="Does this really work? Lets see."

# Tokenising..............................................
# Creating separate words or sentences of the given text

# Importing Libraries used to tokenize
from nltk.tokenize import sent_tokenize,word_tokenize
# Performing Sentence Tokenize.........
sent_arr=sent_tokenize(sample_text_2)
# Performing Word Tokenize.........
# Here we made everything into lower case, but we have lost information due to this. The lowering of case depends upon the usecase.
word_arr=word_tokenize(sample_text_2.lower())
print(sent_arr)
print(word_arr)

# Handling Stopwords and Punctuations.........................................
# Stopwords: Common used in every sentence such as e.g. a, the, in etc.

# Importing Library for stopwords.....
from nltk.corpus import stopwords
# Getting all stopwords for english language...........
stop_words=stopwords.words('english')
# print(stop_words)
# Imprting Library for Punctuations.........
import string
punctuations=list(string.punctuation)
# print(string.punctuation)
stop_words+=punctuations

# Cleaning words(removing stopwords and punctuations)
clean_words=[w for w in word_arr if not w in stop_words]
print(clean_words)

# Stemming...........................................
# Converting similar words to the root word e.g:- play, playing, played to play.

# Importing Library for Stemming.....
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stem_words=['play','playing','played','player',"happy",'happier']
stemmed_words=[ps.stem(w) for w in stem_words]
print(stem_words)
# Stemming has issues as it applies a set of rules to words, and doesn't check whether the word is event correct or not.
# It doesn't provide us with the best result every time, that is get to root word. This can be overcome by Lemmatization.


# Parts of a speech......................................
# This is done to find the actual context of the word, that is it tells us whether the word is adjective, verb, noun etc.
# This is done so as to help us identify the root word for the selected word in terms of the context.

# This is a dataset that is used for the Pos_Tag
from nltk.corpus import state_union
# Importing Libraries for POS_Tag...
from nltk import pos_tag
speech_george_bush_2006=state_union.raw('2006-GWBush.txt')
parts_of_speech=pos_tag(([w for w in word_tokenize(speech_george_bush_2006) if not w in stop_words]))
print(parts_of_speech)

# Lemmatization..........................................
# Converting similar words to the root word e.g:- play, playing, played to play.

# Importing Lemmatization Library.....
from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

a1=lemmatizer.lemmatize('good',pos='n')
a2=lemmatizer.lemmatize('good',pos='a')
a3=lemmatizer.lemmatize('better',pos='a')
a4=lemmatizer.lemmatize('excellent',pos='a')
a5=lemmatizer.lemmatize('paint',pos='n')
a6=lemmatizer.lemmatize('painting',pos='n')
a7=lemmatizer.lemmatize('painting',pos='v')
print('Original Word:',str('good'),'Lemmatized Word:',a1)
print('Original Word:',str('good'),'Lemmatized Word:',a2)
print('Original Word:',str('better'),'Lemmatized Word:',a3)
print('Original Word:',str('excellent'),'Lemmatized Word:',a4)
print('Original Word:',str('paint'),'Lemmatized Word:',a5)
print('Original Word:',str('painting'),'Lemmatized Word:',a6)
print('Original Word:',str('painting'),'Lemmatized Word:',a7)

# Function to Fetch Pos Tag automatically and convert it for lemmatization................
from nltk.corpus import wordnet
def pos_to_wordnet(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('M'):
        return wordnet.MODAL
    elif pos_tag.startswith('R'):
        return wordnet.ADVERB
    else:
        return wordnet.NOUN


