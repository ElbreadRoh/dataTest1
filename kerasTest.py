from keras.layers import Dense
from keras.layers import GRU
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import math

array = []
indices = []
indicesT = []

for x in range(1000):
    array.append(math.sin(x/20))


def regression(dataset, delay=1):
    setA = []
    setB = []
    setC = []
    for i in range(dataset.__len__()):
        if (i + delay) > (dataset.__len__() - 1):
            break

        setA.append(dataset[i: i + delay])
        setB.append(dataset[i + delay])
        setC.append(dataset[i])

    return setA, setB, setC


delayAmt = 400


trainX, trainY, xSet = regression(array, delayAmt)
trainX = np.array(trainX)
trainY = np.array(trainY)

trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)


numps = 0
numps1 = 0


for item in trainX:
    indices.append(numps)
    numps += 1


model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(delayAmt, 1)))
model.add(GRU(100, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(trainX, trainY, epochs=10)


predictions = model.predict(trainX)
inputSet = []
future = []
forwardAmt = 200


for i in range(delayAmt):
    future.append(float(predictions[-(1 + i)]))


for i in range(forwardAmt):
    inputSet = future[i: i + delayAmt]
    inputSet = np.array(inputSet)
    inputSet = inputSet.reshape(1, delayAmt, 1)
    currentStep = model.predict(inputSet)
    future.append(float(currentStep))


model.reset_states()


future = future[delayAmt:]


predIndex = []
indicesMax = 0


for item in indices:
    if indicesMax < item:
        indicesMax = item


for i in range(len(future)):
    predIndex.append(i + indicesMax)


plt.plot(indices, xSet, label='Data set')
plt.plot(predIndex, future, label='Predictions')
plt.title('Sine Wave Prediction Test')
plt.legend(loc='lower right')
plt.xlabel('Time Increments')
plt.ylabel('Sine wave y values')
plt.show()

#pMin = predictA.min()
#pMax = predictA.max()

#predictA = predictA - pMin
#predictA = predictA / (pMax - pMin)

