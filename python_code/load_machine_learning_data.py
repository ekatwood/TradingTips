# cd '' && '/usr/local/bin/python3'  'load_machine_learning_data.py'

import pickle

fileName = '2024-09-14 20:00:00'

def loadData(fileName):
    # Loading the interp1d object from the file
    with open('interpolated_objects/'+fileName+'.pkl', 'rb') as file:
        loaded_interp1d_object = pickle.load(file)

    # Loading the x axis objects from the file
    with open('x_y_axis_objects/x_axis_trade/'+fileName+'.pkl', 'rb') as file:
        loaded_x_axis_trade = pickle.load(file)

    with open('x_y_axis_objects/x_axis_full_trade/'+fileName+'.pkl', 'rb') as file:
        loaded_x_axis_trade_full = pickle.load(file)

    # Loading the x axis objects from the file
    with open('x_y_axis_objects/y_axis_trade/'+fileName+'.pkl', 'rb') as file:
        loaded_y_axis_trade = pickle.load(file)

    with open('x_y_axis_objects/y_axis_full_trade/'+fileName+'.pkl', 'rb') as file:
        loaded_y_axis_trade_full = pickle.load(file)

    return loaded_interp1d_object, loaded_x_axis_trade, loaded_x_axis_trade_full, loaded_y_axis_trade, loaded_y_axis_trade_full

interpolatedObject, x_axis, x_axis_full, y_axis, y_axis_full = loadData(fileName)
