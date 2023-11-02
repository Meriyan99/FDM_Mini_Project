import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('airline_passenger_satisfaction_trained_model.sav', 'rb'))  #read the model


#predict values using loaded model
input_data = (25,562,2,5,5,2,2,2,2,5,3,1,4,2,1,1,1,0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Passenger is dissatisfied or neutral')
else:
  print('The Passenger is satisfied')