import numpy
import pandas
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# create model function and configure input and output neurons (single fully connected hidden layer)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

inputFilePath = "/home/sagar/RadiantLab_Data/ML_Data/test.csv"
modelPath = "/home/sagar/RadiantLab_Data/ML_Data/monthlyCashFlow_100BFinal.h5";
modelJsonPath = "/home/sagar/RadiantLab_Data/ML_Data/monthlyCashFlow_100BFinal.json"
opFilePath = "/home/sagar/RadiantLab_Data/ML_Data/test_op.csv"


# load json and create model
json_file = open(modelJsonPath, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(modelPath)
print("Loaded model from disk")

# load dataset
dataframe = pandas.read_csv(inputFilePath, sep="\t", header=0)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X_Feature_Vector = dataset[:, 0:8]
Y_Output_Vector = dataset[:, 8]
# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_Feature_Vector, Y_Output_Vector, verbose=0)
print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100)

p = loaded_model.predict(X_Feature_Vector, batch_size=10, verbose=0)
f = open(opFilePath, 'w')
#f.write(
#    "pricePerKWh	pricePerTherm	pricePerWattOfPv	kwhPerPvWatt	costAdjustment	incrementalCostAdjustment	annualInterestRate	numberOfPayments	positiveCashFlowCount\n")
for op,y in zip(p, X_Feature_Vector):
    #f.write(str(y)+"\t"+str(int(op))+"\n")
    f.write(str(int(op))+"\n")
f.closed