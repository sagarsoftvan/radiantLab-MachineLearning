import numpy
import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor


# create model function and configure input and output neurons (single fully connected hidden layer)
def baseline_model():
    model = Sequential()
    # Input Layer with 8 Neurons and hidden layer with 8 neurons
    # NN Topology :8 inputs -> [8 Hidden neurons with 1 hidden layer] -> 1 output
    model.add(Dense(8, input_dim=8, init='normal', activation='relu'))
    # Output layer with 1 neuron
    model.add(Dense(1, init='normal'))
    # Compile model with optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# load dataset from CSV into pandas object
dataframe = pandas.read_csv("/home/sagar/RadiantLab_Data/ML_Data/monthlyCashFlow_100B.csv", sep=",", header=0)
dataset = dataframe._values

# split into input (X) and output (Y) variables
X_Feature_Vector = dataset[:, 0:8]
Y_Feature_Vector = dataset[:, 8]

# Print Vector Size
print "X Size", len(X_Feature_Vector)
print "Y Size", len(Y_Feature_Vector)

# Validate Length of Both Vectors (Print error if length is not same)
assert (len(X_Feature_Vector) == len(Y_Feature_Vector))

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# evaluate model with dataset passing with parameters
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
