import numpy as np
import pickle
import streamlit as st
import pandas as pd


#loading the saved model
# loaded_model = pickle.load(open('C:/Users/94775/Desktop/FDM_Project/airline_passenger_satisfaction_trained_model.sav', 'rb'))  #read the model
loaded_model=pd.read_pickle(open('airline_passenger_satisfaction_trained_model.pkl', 'rb'))

#----
# Sidebar with a title and background image
st.sidebar.image("passenger_image.jpg")
# Add CSS to make the image fill the sidebar width
st.sidebar.markdown(
    f'<style>div[data-testid="stSidebar"] div[data-testid="stBlock"] img {{width: 100%;}}</style>',
    unsafe_allow_html=True,
)
st.sidebar.title("How to use?")
# Add simplified instructions
st.sidebar.markdown("1. **Fill the Details:** Fill the detials regarding ratings that has given by the passengers for the services that are provided by the airline.")
st.sidebar.markdown("2. **Get the Satisfaction Level:** Get the satisfaction level of the passenger.")
st.sidebar.markdown("3. **Improve the Services:** Identify the services that should improve to increase the passenger satisfaction level.")
st.sidebar.markdown("4. **Attract passengers more with improved services** ")


st.title('Have a glance of the passenger satisfaction level in brief..')
chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["satisfaction_satisfied","satisfaction_neutral or dissatisfied"])
st.line_chart(chart_data)


# creating a function for Prediction

def passenger_satisfaction_prediction(input_data):
    #take inputs from the user

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return'The Passenger is dissatisfied or neutral'
    else:
        return'The Passenger is satisfied'
  
    
  
def main():
    
    
    # giving a title
    st.title('Predict Satisfaction Level of Airline Passengers')
    st.image("airline.jpg")
    st.subheader('Fill in the details to get the satisfaction level of passengers...')

    Inflight_wifi_service = (
        "--choose--",
        "1",
        "2",
        "3",
        "4",
        "5",
    )
    Departure_Arrival_time_convenient = (
        "--choose--",
        "1",
        "2",
        "3",
        "4",
        "5",
    )
    Ease_of_Online_booking = (
        "--choose--",
        "1",
        "2",
        "3",
        "4",
        "5",
    )
    Online_boarding = (
        "--choose--",
        "1",
        "2",
        "3",
        "4",
        "5",
    )
    Seat_comfort = (
        "--choose--",
        "1",
        "2",
        "3",
        "4",
        "5",
    )
    Inflight_entertainment = (
        "--choose--",
        "1",
        "2",
        "3",
        "4",
        "5",
    )
    On_board_service = (
        "--choose--",
        "1",
        "2",
        "3",
        "4",
        "5",
    )
    Leg_room_service = (
        "--choose--",
        "1",
        "2",
        "3",
        "4",
        "5",
    )
    Baggage_handling = (
        "--choose--",
        "1",
        "2",
        "3",
        "4",
        "5",
    )
    Checkin_service = (
        "--choose--",
        "1",
        "2",
        "3",
        "4",
        "5",
    )
    Inflight_service = (
        "--choose--",
        "1",
        "2",
        "3",
        "4",
        "5",
    )
    Cleanliness = (
        "--choose--",
        "1",
        "2",
        "3",
        "4",
        "5",
    )
    Customer_Type_Loyal_Customer=(
        "--choose--",
        "Loyal customer", 
        "disloyal customer,"
    )

    Type_of_Travel_Business_travel=(
        "--choose--",
        "Personal Travel", 
        "Business Travel",
    )
    
    # getting the input data from the user
    # Age = st.text_input('Age of the Passenger')
    Age = st.slider("Age of the Passenger", 0, 100, 2)
    Flight_Distance = st.text_input('Flight distance of the journey')

    st.subheader('Fill below details according to the ratings that has given by the passengers')

    Inflight_wifi_service = st.selectbox('Satisfaction level of the inflight wifi service',Inflight_wifi_service)
    Departure_Arrival_time_convenient = st.selectbox('Satisfaction level of Departure/Arrival time convenient',Departure_Arrival_time_convenient)
    Ease_of_Online_booking = st.selectbox('Satisfaction level of online booking',Ease_of_Online_booking)
    Online_boarding = st.selectbox('Satisfaction level of online boarding',Online_boarding)
    Seat_comfort = st.selectbox('Satisfaction level of Seat comfort',Seat_comfort)
    Inflight_entertainment = st.selectbox('Satisfaction level of inflight entertainment',Inflight_entertainment)
    On_board_service=st.selectbox('Satisfaction level of On-board service',On_board_service)
    Leg_room_service=st.selectbox('Satisfaction level of Leg room service',Leg_room_service)
    Baggage_handling=st.selectbox('Satisfaction level of baggage handling',Baggage_handling)
    Checkin_service=st.selectbox('Satisfaction level of Check-in service',Checkin_service)
    Inflight_service=st.selectbox('Satisfaction level of inflight service',Inflight_service)
    Cleanliness=st.selectbox('Satisfaction level of Cleanliness',Cleanliness)
    Customer_Type_Loyal_Customer=st.selectbox('Customer type',Customer_Type_Loyal_Customer)
    Type_of_Travel_Business_travel=st.selectbox('Purpose of the flight of the passengers',Type_of_Travel_Business_travel)
    Class_Business=st.text_input('Travel class in the plane(Business)')
    Class_Eco=st.text_input('Travel class in the plane(Eco)')
    
    # code for Prediction
    satisfaction = ''
    
    # creating a button for Prediction
    
    if st.button('Check Satisfaction Level of the Passenger'):
        satisfaction = passenger_satisfaction_prediction([Age,Flight_Distance,Inflight_wifi_service,Departure_Arrival_time_convenient,Ease_of_Online_booking,Online_boarding,Seat_comfort,Inflight_entertainment,On_board_service,Leg_room_service,Baggage_handling,Checkin_service,Inflight_service,Cleanliness,Customer_Type_Loyal_Customer,Type_of_Travel_Business_travel,Class_Business,Class_Eco])

        
        # X = np.array(satisfaction)
        # X[:, 0] = Inflight_wifi_service.transform(X[:,0])
        # X[:, 1] = Departure_Arrival_time_convenient.transform(X[:,1])
        # X[:, 1] = Ease_of_Online_booking.transform(X[:,1])
        # X[:, 1] = Online_boarding.transform(X[:,1])
        # X[:, 1] = Seat_comfort.transform(X[:,1])
        # X[:, 1] = Inflight_entertainment.transform(X[:,1])
        # X[:, 1] = On_board_service.transform(X[:,1])
        # X[:, 1] = Leg_room_service.transform(X[:,1])
        # X[:, 1] = Baggage_handling.transform(X[:,1])
        # X[:, 1] = Checkin_service.transform(X[:,1])
        # X[:, 1] = Inflight_service.transform(X[:,1])
        # X[:, 1] = Cleanliness.transform(X[:,1])
        # X = X.astype(int)

        
        
    st.success(satisfaction)
    
    
   
if __name__ == '__main__':
    main()


# import streamlit as st
# import pickle
# import numpy as np


# def load_model():
#     with open('saved_steps.pkl', 'rb') as file:
#         data = pickle.load(file)
#     return data

# data = load_model()

# regressor = data["model"]
# le_country = data["le_country"]
# le_education = data["le_education"]

# def show_predict_page():
#     st.title("Software Developer Salary Prediction")

#     st.write("""### We need some information to predict the salary""")

#     countries = (
#         "United States",
#         "India",
#         "United Kingdom",
#         "Germany",
#         "Canada",
#         "Brazil",
#         "France",
#         "Spain",
#         "Australia",
#         "Netherlands",
#         "Poland",
#         "Italy",
#         "Russian Federation",
#         "Sweden",
#     )

#     education = (
#         "Less than a Bachelors",
#         "Bachelor’s degree",
#         "Master’s degree",
#         "Post grad",
#     )

#     country = st.selectbox("Country", countries)
#     education = st.selectbox("Education Level", education)

#     expericence = st.slider("Years of Experience", 0, 50, 3)

#     ok = st.button("Calculate Salary")
#     if ok:
#         X = np.array([[country, education, expericence ]])
#         X[:, 0] = le_country.transform(X[:,0])
#         X[:, 1] = le_education.transform(X[:,1])
#         X = X.astype(float)

#         salary = regressor.predict(X)
#         st.subheader(f"The estimated salary is ${salary[0]:.2f}")