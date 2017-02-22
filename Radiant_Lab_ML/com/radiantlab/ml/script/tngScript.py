from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("/home/innoveta/trainingTestDataset.csv", sep=',', lineterminator='\r')
dataset = dataframe.valuess
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]