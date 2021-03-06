import numpy
import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# create model function and configure input and output neurons (single fully connected hidden layer)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

inputFilePath = "/home/sagar/RadiantLab_Data/ML_Data/monthlyCashFlow_100B.csv"
modelPath = "/home/sagar/RadiantLab_Data/ML_Data/monthlyCashFlow_100B.h5";
modelJsonPath = "/home/sagar/RadiantLab_Data/ML_Data/monthlyCashFlow_100B.json"


def baseline_model():
    model = Sequential()
    # Input Layer with 8 Neurons and hidden layer with 8 neurons
    # NN Topology :8 inputs -> [8 Hidden neurons with 1 hidden layer] -> 1 output
    model.add(Dense(8, input_dim=8, init='normal', activation='relu'))
    # Output layer with 1 neuron
    model.add(Dense(1, init='normal'))
    # Compile model with optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')
    #The h5py package is a Pythonic interface to the HDF5 binary data format.
    model.save(modelPath)
    # serialize model to JSON
    model_json = model.to_json()
    with open(modelJsonPath, "w") as json_file:
        json_file.write(model_json)
    return model


# load dataset from CSV into pandas object
dataframe = pandas.read_csv(inputFilePath, sep=",", header=0)
dataset = dataframe._values

# split into input (X) and output (Y) variables
X_Feature_Vector = dataset[:, 0:8]
Y_Output_Vector = dataset[:, 8]

# Print Vector Size
print "X Size", len(X_Feature_Vector)
print "Y Size", len(Y_Output_Vector)

# Validate Length of Both Vectors (Print error if length is not same)
assert (len(X_Feature_Vector) == len(Y_Output_Vector))

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# evaluate model with dataset passing with parameters
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=5, batch_size=5, verbose=0)

# The final step is to evaluate this baseline model. We will use 10-fold cross validation to evaluate the mode
kfold = KFold(n_splits=5, random_state=seed)

# The result reports the mean squared error including the average and standard deviation (average variance)
# across all 5 folds of the cross validation evaluation
results = cross_val_score(estimator, X_Feature_Vector, Y_Output_Vector, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# Now scale the data and check its MSE Error
# Evaluate model with standardized dataset these method scale data based on StandardScaler technique (sklearn.preprocessing.StandardScaler)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=5, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5, random_state=seed)
results = cross_val_score(pipeline, X_Feature_Vector, Y_Output_Vector, cv=kfold)
# Print the MSE for unseen data
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
