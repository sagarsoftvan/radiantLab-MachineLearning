import numpy
import pandas
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt
inputFilePath = "/Users/sagar/Documents/Resources_RadiantLab/ML_Data/monthlyCashFlow_100B.csv"
modelPath = "/Users/sagar/Documents/Resources_RadiantLab/All_Final_PermutationData/monthlyCashFlow_100B_5March.h5";
modelJsonPath = "/Users/sagar/Documents/Resources_RadiantLab/All_Final_PermutationData/monthlyCashFlow_100B_5March.json"

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset from CSV into pandas object
dataframe = pandas.read_csv(inputFilePath, sep=",", header=0)
dataset = dataframe._values

# split into input (X) and output (Y) variables
X_Feature_Vector = dataset[:, 0:8]
Y_Output_Vector = dataset[:, 8]

#plt.plot(dataset)
#plt.show()

# Scale X Feature Vector with mean 0 and std 1
scaler = StandardScaler()
X_Feature_Vector = scaler.fit_transform(X_Feature_Vector)

# Print Vector Size
print "X Size", len(X_Feature_Vector)
print "Y Size", len(Y_Output_Vector)
# Validate Length of Both Vectors (Print error if length is not same)
assert (len(X_Feature_Vector) == len(Y_Output_Vector))

# define base mode and NN Network [8 -> -> 8 -> 6 -> 6 -> 1]
model = Sequential()
model.add(Dense(8, input_dim=8, init='normal', activation='relu'))
model.add(Dense(8, init='normal', activation='relu'))
model.add(Dense(6, init='normal', activation='relu'))
model.add(Dense(6, init='normal', activation='relu'))
model.add(Dense (1, init='normal'))


start = time.time()
print "Start Time %f" %start
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#model.fit(X_Feature_Vector, Y_Output_Vector, nb_epoch=90,batch_size=10, verbose=0)

# Store the best checkpoint weight
filepath="/Users/sagar/Documents/Resources_RadiantLab/ML_Data/weight/weights.best_10march.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit the model [Fit the model with epoch 0 to 150 and store weights which has higher accuracy]
model.fit(X_Feature_Vector, Y_Output_Vector, validation_split=0.33, nb_epoch=2000, batch_size=50, callbacks=callbacks_list, verbose=0)

# evaluate the network
loss, accuracy = model.evaluate(X_Feature_Vector, Y_Output_Vector)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

# make predictions and its accuracy
probabilities = model.predict(X_Feature_Vector)
predictions = [float(round(x)) for x in probabilities]
accuracy = numpy.mean(predictions == Y_Output_Vector)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))

#Store all Data Value

# #The h5py package is a Pythonic interface to the HDF5 binary data format.
model.save(modelPath)
#serialize model to JSON
model_json = model.to_json()
with open(modelJsonPath, "w") as json_file:
       json_file.write(model_json)

end = time.time()
timeTaken = end - start
print "Total Time Taken:"
