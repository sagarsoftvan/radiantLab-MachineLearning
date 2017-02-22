from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# load dataset
dataframe = pandas.read_csv("/home/sagar/RadiantLab_Data/ML_Data/monthlyCashFlow_100B.csv", sep=',', lineterminator='\r',header="0")
dataset = dataframe.valuess
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
print "Done"

# create model
def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=8, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model
