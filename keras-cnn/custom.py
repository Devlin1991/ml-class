import talos as ta
from keras.models import Sequential
from keras.layers import Dense

def minimal():

     x, y = ta.datasets.iris()

     p = {'activation':['relu', 'elu'],
          'optimizer': ['Nadam', 'Adam'],
          'losses': ['logcosh'],
          'hidden_layers':[0, 1, 2],
          'batch_size': [20,30,40],
          'epochs': [10,20]}

     def iris_model(x_train, y_train, x_val, y_val, params):

         model = Sequential()
         model.add(Dense(32, input_dim=8, activation=params['activation']))
         model.add(Dense(1, activation='softmax'))
         model.compile(optimizer=params['optimizer'], loss=params['losses'])

         out = model.fit(x_train, y_train,
                         batch_size=params['batch_size'],
                         epochs=params['epochs'],
                         validation_data=[x_val, y_val])

         return out, model

     scan_object = ta.Scan(x, y, model=iris_model, params=p, grid_downsample=0.1)
     
     return scan_object