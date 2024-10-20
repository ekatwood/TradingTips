# cd '' && '/usr/local/bin/python3'  'getting_crypto_data.py'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import interpolate
from datetime import datetime

# the bot will look at 100 minutes of data
# then, to check if it was a good trade, look at 240 minutes
minutesTrade = 100
minutesFullTrade = 240
crypto = 'BTC'
fileName = 'historical_data/BTC_1min_sample.csv'

# fill arrays with data. x_axis = timestamp, y_axis = open price
def populateArrays(fileName):

    df = pd.read_csv(fileName)

    x_axis = df['timestamp'].values
    y_axis = df['open'].values

    # convert x axis to unix timestamps for interpolation
    date_array = pd.to_datetime(x_axis)

    timestamps = np.array([dt.timestamp() for dt in date_array], dtype=int)

    return timestamps, y_axis


x, y = populateArrays(fileName)

# loop through data, create chunks in 'minutesTrade' length
# exit loop when close to the end of the array
partitionCounter = 0

while(partitionCounter + minutesFullTrade < len(x)):

    currentXAxis = x[partitionCounter:partitionCounter+minutesTrade]
    currentXAxisFullTrade = x[partitionCounter:partitionCounter+minutesFullTrade]
    currentYAxis = y[partitionCounter:partitionCounter+minutesTrade]
    currentYAxisFullTrade = y[partitionCounter:partitionCounter+minutesFullTrade]

    partitionCounter += minutesTrade

    f = interpolate.interp1d(currentXAxis, currentYAxis)

    # serialize interpolated function object in a file
    with open('interpolated_objects/'+str(datetime.fromtimestamp(currentXAxis[0]))+'.pkl', 'wb') as file:
        pickle.dump(f, file)

    # serialize x and y axis objects into files
    with open('x_y_axis_objects/x_axis_trade/'+str(datetime.fromtimestamp(currentXAxis[0]))+'.pkl', 'wb') as file:
        pickle.dump(currentXAxis, file)

    with open('x_y_axis_objects/x_axis_full_trade/'+str(datetime.fromtimestamp(currentXAxisFullTrade[0]))+'.pkl', 'wb') as file:
        pickle.dump(currentXAxisFullTrade, file)

    with open('x_y_axis_objects/y_axis_trade/'+str(datetime.fromtimestamp(currentXAxis[0]))+'.pkl', 'wb') as file:
        pickle.dump(currentYAxis, file)

    with open('x_y_axis_objects/y_axis_full_trade/'+str(datetime.fromtimestamp(currentXAxisFullTrade[0]))+'.pkl', 'wb') as file:
        pickle.dump(currentYAxisFullTrade, file)

    # save price charts
    plt.plot(currentXAxis,currentYAxis)
    plt.title(str(datetime.fromtimestamp(currentXAxis[0])))
    plt.savefig('historical_price_charts/' + str(datetime.fromtimestamp(currentXAxis[0])) + '.png')
    plt.clf()

    # keep log of dates and crypto
    with open('cryptoDates.csv', 'a') as file:
        file.write(crypto + '_' + str(datetime.fromtimestamp(currentXAxis[0])) + ',\n')
