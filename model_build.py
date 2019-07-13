# Importing the libraries
import numpy
import pandas
import nltk
import ssl
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from sklearn import model_selection
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore")

# Importing the dataset

print('Ingesting the data...\n')
train = pandas.read_csv('data/train.txt', delimiter = '\t', header = None, quoting = 3)
train.columns = ['Review', 'Rating']
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('all', quiet = True)

print('Creating the bag of words...\n')

corpus = []
for i in range(0, train.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', train['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model using CountVectorizer

cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = train.iloc[:, 1].values

clf_01 = KNeighborsClassifier()
clf_02 = RandomForestClassifier()
clf_03 = GaussianNB()
clf_04 = BernoulliNB(alpha=0.8)
clf_05 = MultinomialNB(alpha=0.1)
clf_06 = LogisticRegression(C=1.5)
clf_07 = DecisionTreeClassifier()
clf_08 = SVC(kernel="linear")
lr = LogisticRegression()
sclf = StackingClassifier(classifiers = [clf_01, clf_02, clf_03],
                         meta_classifier = lr,
                         use_probas = True,
                         average_probas = False)

print('Performing 5-fold cross validation modelling...\n')

results = []

for clf, label in zip([clf_01, clf_02, clf_03, clf_04, clf_05, clf_06, clf_07, clf_08, sclf],
                     ['KNN',
                     'Random Forest',
                     'Gaussian Naive Bayes',
                     'Bernoulli Naive Bayes',
                     'Multinomial Naive Bayes',
                     'Logistic Regression',
                     'Decision Tree Classifier',
                     'Support Vector Machine',
                     'Stacked Classifier']):
    scores = model_selection.cross_val_score(clf, X, y, cv = 5, scoring = 'accuracy')
    print('Accuracy: %0.6f (+/- %0.2f) [%s]'
         % (scores.mean(), scores.std(), label))
    results.append([label, scores.mean(), scores.std(), clf])

def maximum_accuracy(sequence):
    if not sequence:
        raise ValueError('empty sequence')

    maximum = sequence[0]

    for item in sequence:
        # Compare elements by their weight stored
        # in their second element.
        if item[1] > maximum[1]:
            maximum = item

    return maximum

best_model = maximum_accuracy(results)
best_model[3].fit(X, y)
print('\nThe optimal model was the ' + best_model[0] + ' with an accuracy of ' + str(round(best_model[1], 2)))
print('\n')
print('Ingesting the test data...\n')
test = pandas.read_csv('data/test.txt', delimiter = '\t', header = None, quoting = 3)
test.columns = ['Review']

print('Applying the bag of words...\n')

corpus = []
for i in range(0, test.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', train['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model using CountVectorizer

X_test = cv.transform(corpus).toarray()
test['Predictions'] = best_model[3].predict(X_test)
test['Confidence of Positive (1)'] = [i[1] for i in round(best_model[3].predict_proba(X_test), 2)]
print(test)
test.to_csv('data/test_predictions.txt', index=None, sep='\t')
print('\nResults written out to data/test_predictions.txt')