# Script for  with 10-fold cross validation and display its accruacy and error rate
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
import pandas
from sklearn.preprocessing import StandardScaler


# fix random seed for reproducibility
inputFilePath = "/Users/sagar/Documents/Resources_RadiantLab/ML_Data/80K_Dataset/3april_AllData.tsv"

seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataframe = pandas.read_csv(inputFilePath, sep="\t", header=None)
dataset = dataframe._values

X = dataset[:,0:8]
Y = dataset[:,8]

scaler = StandardScaler()
X = scaler.fit_transform(X)
# Scale X Feature Vector with mean 0 and std 1


# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
    # create models
    model = Sequential  ()
    model.add(Dense(8, input_dim=8, init='normal', activation='relu'))
    model.add(Dense(8, init='normal', activation='relu'))
    #model.add(Dense(8, init='normal', activation='relu'))
    #model.add(Dense(6, init='normal', activation='relu'))
    model.add(Dense(6, init='normal', activation='relu'))
    model.add(Dense(6, init='normal', activation='relu'))
#   model.add(Dense(6, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X[train], Y[train],nb_epoch=1000, batch_size=20, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))