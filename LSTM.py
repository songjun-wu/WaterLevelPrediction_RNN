import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
from sklearn.preprocessing import MinMaxScaler
import datetime


def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def movingAverage(arr, dayForTrain):
    new_arr = np.array([])
    for i in range(arr.shape[0]-dayForTrain):
        new_arr = np.append(new_arr, np.mean(arr[i:i+dayForTrain]))
    return new_arr

def predict(dayForTrain, predictStart, predictEnd, scalar, modelName):

    df1_0 = pd.read_excel(path + 'p_q_t.xlsx', sheet_name='daily', index_col='time')
    df2_0 = pd.read_excel(path+'climate_daily.xlsx', index_col='TIMESTAMP')
    df1_0.index = pd.to_datetime(df1_0.index)
    df2_0.index = pd.to_datetime(df2_0.index)
    predictStart = pd.to_datetime(predictStart)
    predictEnd = pd.to_datetime(predictEnd)

    df1_0 = df1_0[predictStart:predictEnd+datetime.timedelta(days=dayForTrain)][::-1]
    df1_pre = df1_0[['wl_25_corrected']]
    arr1_pre = series_to_supervised(df1_pre.to_numpy().reshape(-1,1), dayForTrain, 1, dropnan=False)
    arr1_pre = arr1_pre[dayForTrain:].values

    df2_pre = df2_0[predictStart:predictEnd+datetime.timedelta(days=dayForTrain)][::-1]
    attributteList = ['AirT_C_Avg', 'Rain_corr_mm_Tot', 'pet']
    df2_pre = df2_pre[attributteList]
    arr2_pre = np.concatenate((movingAverage(df2_pre['AirT_C_Avg'].values, dayForTrain),
                               movingAverage(df2_pre['Rain_corr_mm_Tot'].values, dayForTrain),
                               movingAverage(df2_pre['pet'].values, dayForTrain)), axis=0)
    arr2_pre = arr2_pre.reshape(len(attributteList), -1).T

    arr3_pre = df1_0[['wl_24_corrected']].values
    arr3_pre = arr3_pre[dayForTrain:].reshape(-1,1)
    dataset_pre = np.concatenate((arr2_pre, arr1_pre, arr3_pre), axis=1)

    dataset_pre = scalar.transform(dataset_pre)

    predictX = dataset_pre[:, :-1]
    predictY = np.array([])
    predictX = predictX.reshape(predictX.shape[0], 1, predictX.shape[1])
    #predictX_extended = np.concatenate((predictX, np.full((dayForTrain, 1, predictX.shape[2]),np.nan)), axis=0)

    model = keras.models.load_model(modelName)
    for i in range(predictX.shape[0]):
        predict_tmp = model.predict(predictX[i, :, :].reshape(1,1,-1))
        predictY = np.append(predictY, predict_tmp)

    #plt.figure()
    #plt.plot(np.arange(0, predictX.shape[0], 1), predictY)
    #plt.show()

    tmp = np.concatenate((predictX, predictY.reshape(-1, 1, 1)), axis=2)
    tmp = tmp[:, 0, :]
    tmp = scalar.inverse_transform(tmp)
    tmp = tmp[::-1]
    tmp = tmp[:,-1]
    tmp[tmp<0] = 0

    np.savetxt('LSTM_predict.txt', tmp)

    return predictX, predictY

def createDatasets(path, dayForTrain, trainTeststart, trainTestend, trainTestRatio):
    df1_0 = pd.read_excel(path+'p_q_t.xlsx', sheet_name='daily', index_col='time')
    df2_0 = pd.read_excel(path+'climate_daily.xlsx', index_col='TIMESTAMP')

    df1_0.index = pd.to_datetime(df1_0.index)
    df2_0.index = pd.to_datetime(df2_0.index)
    trainTeststart = pd.to_datetime(trainTeststart)
    trainTestend = pd.to_datetime(trainTestend)

    df1_0 = df1_0[trainTeststart:trainTestend+datetime.timedelta(days=dayForTrain)][::-1]
    df1 = df1_0['wl_25_corrected']
    arr1 = series_to_supervised(df1.to_numpy().reshape(-1,1), dayForTrain, 1, dropnan=False)
    arr1 = arr1[dayForTrain:].values


    df2 = df2_0[trainTeststart:trainTestend+datetime.timedelta(days=dayForTrain)][::-1]
    attributteList = ['AirT_C_Avg', 'Rain_corr_mm_Tot', 'pet']
    df2 = df2[attributteList]
    arr2 = np.concatenate((movingAverage(df2['AirT_C_Avg'].values, dayForTrain), movingAverage(df2['Rain_corr_mm_Tot'].values, dayForTrain), movingAverage(df2['pet'].values, dayForTrain)), axis=None)
    arr2 = arr2.reshape(len(attributteList), -1).T

    arr3 = df1_0['wl_24_corrected'].values
    arr3 = arr3[dayForTrain:].reshape(-1,1)

    dataset = np.concatenate((arr2, arr1, arr3), axis=1)

    n_train = int(trainTestRatio*np.shape(dataset)[0])

    train = dataset[:n_train, :]
    test  = dataset[n_train:, :]

    scaler = MinMaxScaler(feature_range=(0,1))
    scalar = scaler.fit(train)
    train = scalar.transform(train)
    test  = scalar.transform(test)


    trainX, trainY = train[:,:-1], train[:,-1]
    testX , testY  = test[:, :-1], test[:, -1]
    trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
    testX  = testX.reshape(testX.shape[0] , 1, testX.shape[1] )

    return trainX, testX, trainY, testY, scalar

def Train(trainX, testX, trainY, testY, ifTrain, modelName):
    if ifTrain:
        model = Sequential()
        model.add(LSTM(80, input_shape = (trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        history = model.fit(trainX, trainY, epochs=10000, validation_data=(testX, testY), verbose=2, shuffle=False)
        model.save('LSTM.model')
    else:
        model = keras.models.load_model(modelName)

    predict_trainY = model.predict(trainX)
    predict_testY = model.predict(testX)

    np.savetxt('LSTM_train.txt', np.concatenate((trainY[::-1], predict_trainY[::-1]), axis=None))
    np.savetxt('LSTM_test.txt' , np.concatenate((testY[::-1] , predict_testY [::-1]), axis=None))

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(15, 5))
    ax[0].plot(np.arange(0, trainX.shape[0], 1), trainY)
    ax[0].plot(np.arange(0, trainX.shape[0], 1), predict_trainY)
    ax[1].plot(np.arange(0, testX.shape[0], 1), testY)
    ax[1].plot(np.arange(0, testX.shape[0], 1), predict_testY)
    plt.show()

    return trainX, trainY


path = r'C:\Users\songjunwu\Downloads\3/data/'
#path = r'D:\OneDrive\Phd\WorkPackages\3\data/'


dayForTrain = 20
modelName = 'LSTM_good4.model'
trainTeststart = '2021-03-24'
trainTestend = '2021-12-28'
trainTestRatio = 0.7
predictStart = '2020-01-08'
predictEnd = '2021-12-28'

ifTrain = False



trainX, testX, trainY, testY, scalar = createDatasets(path, dayForTrain, trainTeststart, trainTestend, trainTestRatio)

trainX, trainY = Train(trainX, testX, trainY, testY, ifTrain, modelName)

if ifTrain:
    modelName = 'LSTM.model'

predictX, predictY = predict(dayForTrain, predictStart, predictEnd, scalar, modelName)


fig, ax = plt.subplots(4, 3, sharex=True, sharey=False, figsize=(7, 7))
for i in range(4):
    ax[i,0].plot(np.arange(trainX.shape[0]),trainX[:,0,i])
    ax[i,1].plot(np.arange(predictX.shape[0]), predictX[:, 0, i])
    ax[i,2].plot(np.arange(predictY.shape[0]), predictY)
    ax[i,2].plot(np.arange(trainY.shape[0]), trainY)
plt.show()


