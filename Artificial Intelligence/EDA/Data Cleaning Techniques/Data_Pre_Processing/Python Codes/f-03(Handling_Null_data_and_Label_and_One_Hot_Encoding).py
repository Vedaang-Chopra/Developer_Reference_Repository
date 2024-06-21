# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('D:\Artificial Intelligence\Code Repository\Machine_Learning_Code_Repository\All_External_CSV_Datasets_Used\Data.csv')
# print(dataset)

# Handling Missing Data using Imputer Class..................................
# For missing data we can either delete the data, which means losing information, or fill it with a value. The value could me mean, zero depending upon the dataset.
# Note:-Empty Data/ Nan can only be filled for continuous valued Data.

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
print(X)

# Taking care of missing data using Imputer Class...........
from sklearn.preprocessing import Imputer

# Imputer Class helps to pre-process data. It has multiple parameters such as:
# missing_values: This parameter helps us to identify which parameters to fill.
# strategy: The value selected by missing_values is replaced by this value. Could be the mean, or 0 etc. Default is mean.
# axis : Specify either row or column wise
# Imputer object............

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# Here Imputer object is made to fit and then transform the data. The imputer doesn't need to fitt on all the data, but rather the selected values only.
imputer = imputer.fit(X[:, 1:3])
# Replacing the missing data with the new values...............
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)


# Encoding categorical data...........
# Encoding the Categorical Data is done because text data as classes can't be processed by computer as computer processes numbers.
# Thus text classes are converted to number classes.

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Label Encoder is used to encode the text classes that are present in the dataset and transform them into number classes.
# That is instead of classes being identified as texxt they will be identified as numbers.
# This can be done both on the Y output and X feature columns.
# Note:; For each unique x feature encoding has to be done separately.
labelencoder_X = LabelEncoder()
# Here the label encoder identifies different types of classes and them identifies each unique class as number. Then it transforms the current dataset to it.
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
print(X)
# Note:-
# Label encoding has a problem. If classes have no relation, say a column had counteries as classes then each country would be a unique number.
# But as each number has a magnitude thus a weight would be identified with each country which shouldn't have happened.
# For cases like small, medium, large in regression problems this could work.

# SO ONE HOT ENCODING IS DONE.................
# Instead of encoding each class as a number, we create multiple columns, that is assume a class has 3 features, then instead of 0,1,2 as classes we would do:
# For class 0: [1,0,0]
# For class 1: [0,1,0]
# For class 2: [0,0,1]
# So the columns are increased


onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)