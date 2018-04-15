# Import related packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from collections import Counter


# Data cleaning and data exploration
# Load csv document
# Show top 5 observations

raw_data=pd.read_csv('Campaign_table0.csv')
raw_data.head()

# Fill empty and NaNs values with NaN
data=raw_data.fillna(np.NaN)
# Check null and missing values
data.isnull().sum()

# Summerize the data, look for possible outliers
data.describe()

# Plot charts

Plt.hist(data["Age"])
