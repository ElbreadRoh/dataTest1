import matplotlib.pyplot as plt
import numpy as np
import math
from keras.layers import Dense
from keras.layers import GRU
from keras.models import Sequential
from keras.layers import Dropout


plt.clf()

reader = open(r'C:\Users\hydra\Documents\fluxData.txt', 'r')

radioEmission = []
EmTimeline = []
indices = []  # redundant but oh well
indexValue = 0
valuesDate1 = []
valuesDate2 = ['1950', '1960', '1970', '1980', '1990', '2000', '2010']

# F10.7 Radio Emission
for item in reader:
    line = item.rstrip()
    year = int(line[0:8])

    radioData = (line[12:17])
    if radioData != '':
        radioData = float(radioData)
        EmTimeline.append(year)
        radioEmission.append(radioData)
        indexValue += 1
        indices.append(indexValue)
        if year in (19500101, 19600102, 19700101, 19800101, 19900101, 20000101, 20100101):
            valuesDate1.append(indexValue)
    else:
        EmTimeline.append(year)
        radioEmission.append(None)
        indexValue += 1
        indices.append(indexValue)
        if year in (19500101, 19600102, 19700101, 19800101, 19900101, 20000101, 20100101):
            valuesDate1.append(indexValue)


for i in range(len(radioEmission)):
    if EmTimeline[i] == 19800101:
        print(i)

#plt.plot(indices, radioEmission, c='red')

# Too heavy
#plt.bar(indices, radioEmission)
#plt.xlabel('Time (in years)')
#plt.ylabel('F10.7 Radio Emission (s.f.u)')
#plt.title('Sun Activity through F10.7 Emission')

#plt.xticks(valuesDate1, valuesDate2)
#plt.show()
# plt.clf()

# Xray flux


xRayRawFluxValues = []
xRayConvertedValues = []
xRayTimeline = []
xRayConvertedTimeline = []
indices2 = []
possibleDates = []
indexValue2 = 0

reader2 = open(r'C:\Users\hydra\Documents\xRayCombined.txt', 'r')

# Reader
for item in reader2:
    line = item.rstrip()
    year2 = line[5:11]
    xRayValue = line[59:63]

    if float(line[5:7]) == 17:
        break

    xRayRawFluxValues.append(xRayValue)
    xRayTimeline.append(year2)
    indexValue2 += 1
    indices2.append(indexValue2)


# Converter
def xRayUnitConversion(arrayInput, isInconsistent, arrayOutput):
    index = 0

    for item in arrayInput:
        index += 1
        if item[1:4] != '   ' and item[0:1] != '':
            if item[1:2] != ' ':
                if item[0:1] == 'A':
                    item = float(item[1:4]) * (10 ** -8)
                elif item[0:1] == 'B':
                    item = float(item[1:4]) * (10 ** -7)
                elif item[0:1] == 'C':
                    if float(item[1:4]) == 0:
                        item = 0.9 * (10 ** -6)
                    else:
                        item = float(item[1:4]) * (10 ** -6)
                elif item[0:1] == 'M':
                    item = float(item[1:4]) * (10 ** -5)
                elif item[0:1] == 'X':
                    item = float(item[1:4]) * (10 ** -4)
                elif item[0:4] == '<A1.':
                    item = float(item[2:4]) * (10 ** -9)
            elif item[1:2] == ' ':
                if item[0:1] == 'A':
                    item = float(item[2:4]) * (10 ** -8)
                elif item[0:1] == 'B':
                    item = float(item[2:4]) * (10 ** -7)
                elif item[0:1] == 'C':
                    if float(item[2:4]) == 0:
                        item = 0.9 * (10 ** -6)
                    else:
                        item = float(item[2:4]) * (10 ** -6)
                elif item[0:1] == 'M':
                    item = float(item[2:4]) * (10 ** -5)
                elif item[0:1] == 'X':
                    item = float(item[2:4]) * (10 ** -4)

            if index > 3542 and isInconsistent:
                item = item / 10

            arrayOutput.append(item)
        else:
            arrayOutput.append(0)


xRayUnitConversion(xRayRawFluxValues, True, xRayConvertedValues)


# Modified Timeline
#format dates into year/month/date format, ex: 19750102 = jan 2, 1975
for item in xRayTimeline:
    if item[0:1] == '0' or item[0:1] == '1':
        item = int("20" + item)
    else:
        item = int('19' + item)

    xRayConvertedTimeline.append(item)


# Consistent Timeline Maker
def timelineMaker(yearStart, yearEnd, yearSet):
    while yearStart <= yearEnd:
        yearSet.append(yearStart)

        yearStart += 1
        if int(str(yearStart)[6:]) > 31 and int(str(yearStart)[4:6]) in (1, 3, 5, 7, 8, 10, 12):
            yearStart -= 32
            yearStart += 101
        elif int(str(yearStart)[6:]) > 30 and int(str(yearStart)[4:6]) in (4, 6, 9, 11):
            yearStart -= 31
            yearStart += 101
        elif int(str(yearStart)[0:4]) % 4 == 0 and int(str(yearStart)[4:6]) == 2:
            if int(str(yearStart)[6:]) > 29:
                yearStart -= 30
                yearStart += 101
        elif int(str(yearStart)[0:4]) % 4 != 0 and int(str(yearStart)[4:6]) == 2:
            if int(str(yearStart)[6:]) > 28:
                yearStart -= 29
                yearStart += 101

        if int(str(yearStart)[4:]) == 1301:
            yearStart -= 1200
            yearStart += 10000


timelineMaker(19750101, 20161231, possibleDates)

# Lining up the data w/ consistent timeline
indices3 = []
fluxTemp = 0
xRayTestValues = []


# Xrayconvertectimeline compared with possibleDates (both ints)

def timeSetCreation(yearSet, indexOutputSet, timeline, dataset, setOutput, yearLimit):
    indexFinder = 0
    index = 0
    for item in yearSet:
        indexOutputSet.append(index)
        index += 1

        if timeline[indexFinder] > item:
            setOutput.append(None)

        # chooses the largest value in one day if there are multiple values
        elif timeline[indexFinder] == item:
            xRayFluxTop = 0
            while timeline[indexFinder] <= yearLimit and timeline[indexFinder] <= item:
                if xRayFluxTop < dataset[indexFinder]:
                    xRayFluxTop = dataset[indexFinder]
                if indexFinder == timeline.__len__() - 1:
                    break

                indexFinder += 1

            setOutput.append(xRayFluxTop)
        else:
            setOutput.append(None)


timeSetCreation(possibleDates, indices3, xRayConvertedTimeline, xRayConvertedValues, xRayTestValues, 20161231)

reader3 = open(r'C:\Users\hydra\Documents\xRayValFunction.txt', 'r')

xRayValDailyValues = []
xRayValDailyConvertedValues = []
xRayValRawValues = []
xRayValDates = []
xRayFullSet = []
xRayValElement = 0
valueAvailable = False
findBackgroundValue = False
bgValueAvailable = False

xRayConvertedValTimeline = []
indices4 = []
possibleDates2 = []
xRayValValues = []


for item in reader3:
    line = item.rstrip()

    #processes xray values after all values for the day are received
    if line[0:17] == 'B.  Proton Events':
        valueAvailable = False
        xRayUnitConversion(xRayValDailyValues, False, xRayValDailyConvertedValues)
        valValueMax = 0
        for item2 in xRayValDailyConvertedValues:
            if item2 > valValueMax:
                valValueMax = item2
        if valValueMax == 0:
            findBackgroundValue = True
        else:
            xRayValRawValues.append(valValueMax)

        xRayValDailyValues.clear()
        xRayValDailyConvertedValues.clear()

    #accepts xray values from the dataset
    if valueAvailable:
        xRayValDailyValues.append(line[29:33])

    #accepts background values from the dataset (only if there are no prominent xray values)
    if bgValueAvailable:
        xRayValDailyValues.append(line[54:58])
        xRayUnitConversion(xRayValDailyValues, False, xRayValDailyConvertedValues)
        for item3 in xRayValDailyConvertedValues:
            xRayValElement = item3

        xRayValRawValues.append(xRayValElement)
        xRayValDailyValues.clear()
        xRayValDailyConvertedValues.clear()

        xRayValElement = 0
        findBackgroundValue = False
        bgValueAvailable = False

    if line[0:11] == 'SGAS Number':
        xRayValDates.append(line[35:46])

    if line[0:20] == 'A.  Energetic Events':
        reader3.__next__()
        valueAvailable = True

    if line[0:13] == 'D.  Stratwarm' and findBackgroundValue:
        reader3.__next__()
        bgValueAvailable = True


#format dates into year/month/date format, ex: 19750102 = jan 2, 1975
for item in xRayValDates:
    output = int(item[7:11]) * 10000
    output = output + int(item[0:2])

    if item[3:6] == 'Jan':
        output = output + 100
    if item[3:6] == 'Feb':
        output = output + 200
    if item[3:6] == 'Mar':
        output = output + 300
    if item[3:6] == 'Apr':
        output = output + 400
    if item[3:6] == 'May':
        output = output + 500
    if item[3:6] == 'Jun':
        output = output + 600
    if item[3:6] == 'Jul':
        output = output + 700
    if item[3:6] == 'Aug':
        output = output + 800
    if item[3:6] == 'Sep':
        output = output + 900
    if item[3:6] == 'Oct':
        output = output + 1000
    if item[3:6] == 'Nov':
        output = output + 1100
    if item[3:6] == 'Dec':
        output = output + 1200

    xRayConvertedValTimeline.append(output)

# Data corrections
xRayValRawValues[235] = 0.0
xRayValRawValues[1009] = 0.0
xRayValRawValues[1710] = 1.32 * (10 ** -8)


timelineMaker(20170101, 20220118, possibleDates2)
timeSetCreation(possibleDates2, indices4, xRayConvertedValTimeline, xRayValRawValues, xRayValValues, 20220118)

indices5 = []
for item in indices4:
    indices5.append(item + indices3.__len__())

indices6 = indices3 + indices5
xRayFullSet = xRayTestValues + xRayValValues
possibleDates3 = possibleDates + possibleDates2


#Cutting off the data set
#This is due to the large number of blanks between 1975-1980
possibleDates3 = possibleDates3[1826:]
indices6 = indices6[1826:]
xRayFullSet = xRayFullSet[1826:]

for x in range(indices6.__len__()):
    indices6[x] = indices6[x] - 1826

#Filling in data gaps
xRayFullSet[4999] = None


for item in indices6:
    if xRayFullSet[item] is None:
        xRayFullSet[item] = 0.000000001
    elif xRayFullSet[item] == 0:
        xRayFullSet[item] = 0.000000001


#timeline broad (for display purposes)
valuesDate3 = []
valuesDate4 = ['1980', '1985', '1990', '1995', '2000', '2005', '2010', '2015', '2020']

for item in indices6:
    if possibleDates3[item] in (19800101, 19850101, 19900101, 19950101, 20000101, 20050101, 20100101, 20150101, 20200101):
        valuesDate3.append(item)


#timeline specific (for testing purposes)
valuesDate5 = []
valuesDate6 = []
valueDate6Tick = 19750101

while valueDate6Tick < 20230101:
    valuesDate6.append(valueDate6Tick)
    valueDate6Tick = valueDate6Tick + 10000

for item in indices6:
    if possibleDates3[item] in valuesDate6:
        valuesDate5.append(item)

#Y-axis specification
##Conversions
##A 10^-8 watts/m^2
##B 10^-7
##C 10^-6
##M 10^-5
##X 10^-4

#valuey1 = [10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3]
#valuey2 = ['A', 'B', 'C', 'M', 'X', 'X+']

valuey1 = []
valuey2 = []


bad1 = []
bad2 = []
ib = []
ib2 = []


reader5 = open(r'C:\Users\hydra\Desktop\ml1.txt', 'r')

for item in reader5:
    line = item.rstrip()
    bad1.append(float(line))


reader45 = open(r'C:\Users\hydra\Desktop\ml2.txt', 'r')

for item in reader45:
    line = item.rstrip()
    bad2.append(float(line))


for i in range(len(bad1)):
    ib.append(15359 + i)


for i in range(len(bad2)):
    ib2.append(15359 + i)


radioEmission = radioEmission[12053:]


for i in range(len(bad1)):
    bad1[i] = (bad1[i] * (0.0028 - 0.000000001)) + 0.000000001


for i in range(len(bad2)):
    bad2[i] = (bad2[i] * (0.0028 - 0.000000001)) + 0.000000001

xRayFullSet = np.array(xRayFullSet)


for i in range(len(radioEmission)):
    if radioEmission[i] is None:
        radioEmission[i] = (radioEmission[i + 1] + radioEmission[i - 1]) / 2


radioEmission = np.array(radioEmission)

xRayFullSet = (xRayFullSet * (radioEmission.max() / xRayFullSet.max())) + radioEmission.min()

plt.plot(indices6, xRayFullSet, c='blue', alpha=0.3, label='X-ray flux (W/m^2)')
#plt.plot(ib2, bad2, c='orange', label='Prediction')
plt.plot(radioEmission, c='red', alpha=0.3, label='F10.7 Radio Emission (10^-22 * W/m^2)')
plt.xlabel('Time (in years)')
plt.legend(loc='upper right')
plt.ylabel('Apparent magnitude of fluxes (not to scale)', labelpad=0)
plt.title('Comparisons of X-ray and Radio Emission fluxes over time')

plt.xticks(valuesDate3, valuesDate4)
plt.yticks(valuey1, valuey2)

plt.show()



# 17197 total set
# 15338 cut set

# 10535 biggest in original
# 8708 biggest in cut
#max value = 0.0028
#min value = 0.000000006

#Normalization
xRayFullSet = np.array(xRayFullSet)
xrns = xRayFullSet





for i in range(xrns.__len__()):
    xrns[i] = (xRayFullSet[i] - 0.000000001) / (0.0028 - 0.000000001)


radioEmission = radioEmission[300:]
indices = indices[300:]

lowBound = 0
upBound = 0

for i in range(len(radioEmission)):
    if radioEmission[i] is None:
        for k in range(len(radioEmission) - i):
            if not(radioEmission[i + k] is None):
                upBound = k
                break

        for k in range(i):
            if not(radioEmission[i - k] is None):
                lowBound = k
                break

        radioEmission[i] = (radioEmission[i-lowBound] + radioEmission[i+upBound]) / 2


rens = np.array(radioEmission)
rens = rens[5000:]


rMax = rens.max()
rMin = rens.min()


for i in range(len(rens)):
    rens[i] = (rens[i] - rMin) / (rMax - rMin)



#Smoothening/averaging
window = 50
smoothSet = []

for i in range(len(xrns) - window):
    average = 0
    for k in range(window):
        average += xrns[k + i]

    average = average / window
    smoothSet.append(average)



#regression
def regression(dataset, delay=1):
    setA = []
    setB = []
    setC = []
    for i in range(dataset.__len__()):
        if (i + delay) > (dataset.__len__() - 1):
            break

        setA.append(dataset[i:i + delay])
        setB.append(dataset[i + delay])
        setC.append(dataset[i])

    return setA, setB, setC


#trainSet = xrns[:13515]
#testSet = xrns[13515:]

delayAmt = 4015


#x_train, y_train, xSet = regression(trainSet, delayAmt)
#x_test, y_test, ySet = regression(testSet, delayAmt)
x_train, y_train, xSet = regression(xrns, delayAmt)


#Reshaping
x_train = np.array(x_train)
y_train = np.array(y_train)
#x_test = np.array(x_test)
#y_test = np.array(y_test)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)



#Machine Learning
model = Sequential()
model.add(GRU(100, input_shape=(delayAmt, 1), return_sequences=True, activation='elu', kernel_initializer='lecun_normal'))
model.add(GRU(100, return_sequences=False))
model.add(Dense(1, activation='elu', kernel_initializer='lecun_normal'))
model.compile(optimizer='adam', loss='mse')
history = model.fit(x_train, y_train, epochs=10)


#Antiskewing
#these are originally arrays


for i in range(1):
    trainPredict = trainPredict.reshape(trainPredict.shape[0], trainPredict.shape[1], 1)
    trainPredict = model.predict(trainPredict)

    #minA = trainPredict.min()
    #maxA = trainPredict.max()

    #trainPredict = trainPredict - minA
    #trainPredict = trainPredict / (maxA - minA)

    skewSet[i + 1] += float(trainPredict.max() - trainPredict.min())



#Validation
trainPredict = model.predict(x_train)
#testPredict = model.predict(x_test)

trpMin = trainPredict.min()
trpMax = trainPredict.max()
#tepMin = testPredict.min()
#epMax = testPredict.max()


#testSet = np.array(testSet)
#tetMax = testSet.max()
#tetMin = testSet.min()
#testSet = (testSet - tetMin) / (tetMax - tetMin)


trainPredict = trainPredict - trpMin
trainPredict = trainPredict / (trpMax - trpMin)
#testPredict = testPredict - tepMin
#testPredict = testPredict / (tepMax - tepMin)


#Prediction
inputSet = []
futureSet = []
forwardAmt = 3650

for i in range(delayAmt):
    futureSet.append(float(trainPredict[-(1 + i)]))


for i in range(forwardAmt):
    inputSet = futureSet[i: i + delayAmt]
    inputSet = np.array(inputSet)
    inputSet = inputSet.reshape(1, delayAmt, 1)
    currentStep = model.predict(inputSet)
    futureSet.append(float(currentStep))


futureSet = futureSet[delayAmt:]


predIndex = []
indicesMax = 0
indices7 = []

for i in range(len(x_train)):
    indices7.append(i)


for item in indices7:
    if indicesMax < item:
        indicesMax = item


for i in range(len(futureSet)):
    predIndex.append(i + indicesMax)


f = open("file2.txt", "a")


for item in futureSet:
    f.write(str(item))
    f.write('\n')


plt.plot(indices7, xSet, c='red')
plt.plot(predIndex, futureSet, c='blue')

plt.show()



