import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime
st.set_page_config(layout="wide")


FUEL_PRICE_PER_LITER = 105

model = joblib.load('fuel_efficiency_gb_model.joblib')
model_features = joblib.load('feature_names.joblib')
data = pd.read_csv("C:/Users/psrig/Downloads/Fuel_Consumption_2000-2022.csv")

makes = ['ACURA', 'ALFA ROMEO', 'ASTON MARTIN', 'AUDI', 'BENTLEY', 'BMW', 'BUGATTI', 'BUICK', 'CADILLAC', 'CHEVROLET',
         'CHRYSLER', 'DAEWOO', 'DODGE', 'FERRARI', 'FIAT', 'FORD', 'GENESIS', 'GMC', 'HONDA', 'HUMMER', 'HYUNDAI',
         'INFINITI', 'ISUZU', 'JAGUAR', 'JEEP', 'KIA', 'LAMBORGHINI', 'LAND ROVER', 'LEXUS', 'LINCOLN', 'MASERATI',
         'MAZDA', 'MERCEDES-BENZ', 'MINI', 'MITSUBISHI', 'NISSAN', 'OLDSMOBILE', 'PLYMOUTH', 'PONTIAC', 'PORSCHE',
         'RAM', 'ROLLS-ROYCE', 'SAAB', 'SATURN', 'SCION', 'SMART', 'SRT', 'SUBARU', 'SUZUKI', 'TOYOTA', 'VOLKSWAGEN',
         'VOLVO']

vehicle_classes = ['COMPACT', 'FULL-SIZE', 'MID-SIZE', 'MINICOMPACT', 'MINIVAN', 'PICKUP TRUCK - SMALL',
                   'PICKUP TRUCK - STANDARD', 'SPECIAL PURPOSE VEHICLE', 'STATION WAGON - MID-SIZE',
                   'STATION WAGON - SMALL', 'SUBCOMPACT', 'SUV', 'SUV - SMALL', 'SUV - STANDARD', 'TWO-SEATER',
                   'VAN - CARGO', 'VAN - PASSENGER']

transmissions = ['A10', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'AM5', 'AM6', 'AM7', 'AM8', 'AM9', 'AS10', 'AS4',
                 'AS5', 'AS6', 'AS7', 'AS8', 'AS9', 'AV', 'AV1', 'AV10', 'AV6', 'AV7', 'AV8', 'M4', 'M5', 'M6', 'M7']

fuel_types = {
    'X': 'Regular gasoline',
    'Z': 'Premium gasoline',
    'D': 'Diesel',
    'E': 'Ethanol (E85)',
    'N': 'Natural Gas'
}

page = st.sidebar.radio("Go to", ["Prediction", "Dashboard"])

if page == "Prediction":
    st.title("Fuel Efficiency Prediction Dashboard")

st.header("Enter Vehicle Specifications:")
engine_size = st.slider("Engine Size (L)", min_value=0.8, max_value=8.4, value=3.0, step=0.1)
cylinders = st.slider("Cylinders", min_value=2, max_value=16, value=6, step=1)
fuel_consumption = st.slider("Fuel Consumption (Comb) (L/100 km)", min_value=3.6, max_value=26.1, value=12.0)
highway_fuel_consumption = st.slider("Highway Fuel Consumption (L/100 km)", min_value=3.2, max_value=20.9, value=8.5)
make = st.selectbox("Make", options=makes)
vehicle_class = st.selectbox("Vehicle Class", options=vehicle_classes)
transmission = st.selectbox("Transmission", options=transmissions)

fuel_type_display = st.selectbox("Fuel Type", options=fuel_types.values())
fuel_type = [k for k, v in fuel_types.items() if v == fuel_type_display][0]

input_data = {
    'ENGINE SIZE': [engine_size],
    'CYLINDERS': [cylinders],
    f'MAKE_{make}': 1,
    f'VEHICLE CLASS_{vehicle_class}': 1,
    f'TRANSMISSION_{transmission}': 1,
    f'FUEL_{fuel_type}': 1
}

input_df = pd.DataFrame(input_data)
input_df = input_df.reindex(columns=model_features, fill_value=0)

if 'predictions' not in st.session_state:
    st.session_state.predictions = []

if st.button("Predict Fuel Consumption"):
    prediction = model.predict(input_df)[0]
    estimated_cost = prediction * FUEL_PRICE_PER_LITER
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Store prediction and estimated cost in session state
    new_prediction = {'Timestamp': timestamp, 'Fuel Consumption': prediction, 'Estimated Cost': estimated_cost}
    st.session_state.predictions.append(new_prediction)

    st.write(f"Predicted Combined Fuel Consumption: {prediction:.2f} L/100 km")
    st.write(f"Estimated Fuel Cost for 100 km: ₹{estimated_cost:.2f}")

    filtered_data = data[(data['MAKE'] == make) & (data['VEHICLE CLASS'] == vehicle_class) &
                         (data['CYLINDERS'] == cylinders) & (data['TRANSMISSION'] == transmission) &
                         (data['FUEL'] == fuel_type)]
    
    if not filtered_data.empty:
        fig = px.histogram(
            filtered_data,
            x='COMB (L/100 km)',
            title='Fuel Consumption of Similar Vehicles',
            labels={'COMB (L/100 km)': 'Combined Fuel Consumption (L/100 km)'}
        )
        st.plotly_chart(fig)
    else:
        st.write("No similar vehicles found in the dataset for comparison.")

    model_data = data[(data['MAKE'] == make) & (data['VEHICLE CLASS'] == vehicle_class)]
    if not model_data.empty:
        max_fuel_consumption = model_data['COMB (L/100 km)'].max()
        st.session_state['max_fuel_consumption'] = max_fuel_consumption

        if prediction > max_fuel_consumption:
            st.error("WARNING: Your car is consuming more fuel than the highest recorded for similar vehicles.", icon="⚠️")
            st.write("**Fuel-Saving Tips:** Avoid rapid acceleration, maintain optimal tire pressure, and reduce excess weight.")
            st.write("### High Fuel Consumption")
            st.write("#### Causes:")
            st.write("""
                - **Under-inflated Tires**: Increases rolling resistance.
                - **Dirty Air Filter**: Reduces air intake efficiency, causing the engine to burn more fuel.
                - **Worn Spark Plugs**: Inefficient combustion leads to more fuel usage.
                - **Bad Oxygen Sensor**: Misreading fuel-air mixture data can lead to excess fuel usage.
                - **Aggressive Driving Habits**: Rapid acceleration and heavy braking consume more fuel.
                - **Excessive Idling**: Prolonged idling burns fuel without moving the vehicle.
            """)
            st.write("#### Symptoms:")
            st.write("""
                - Decreased miles per gallon (mpg).
                - Frequent refueling required.
                - Poor engine performance.
            """)
            st.write("#### Repairs:")
            st.write("""
                - **Inflate Tires**: Check and maintain optimal tire pressure.
                - **Replace Air Filter**: Install a new air filter if it's clogged or dirty.
                - **Change Spark Plugs**: Replace worn spark plugs for efficient combustion.
                - **Check Oxygen Sensors**: Replace faulty sensors to improve fuel-air ratio.
                - **Drive Smoothly**: Avoid rapid acceleration and braking.
                - **Reduce Idling**: Turn off the engine when parked.
            """)

if page == "Dashboard":
    st.title("Prediction History Dashboard")

    if 'predictions' in st.session_state and st.session_state.predictions:
        predictions_df = pd.DataFrame(st.session_state.predictions)
        max_fuel_consumption = st.session_state.get('max_fuel_consumption', predictions_df['Fuel Consumption'].max())

        # Compute total predictions, normal conditions, and warning conditions
        total_predictions = len(predictions_df)
        normal_conditions = sum(predictions_df['Fuel Consumption'] <= max_fuel_consumption)
        warning_conditions = total_predictions - normal_conditions

        # Display summary statistics
        st.markdown("### Summary Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", total_predictions)
        col2.metric("Normal", normal_conditions)
        col3.metric("Warning", warning_conditions)

        # Display recent predictions
        st.markdown("### Recent Predictions")
        st.write(predictions_df)

        # Charts

        fig2 = px.line(predictions_df, x="Timestamp", y="Fuel Consumption", title="Fuel Consumption Over Time")
        st.plotly_chart(fig2)

        fig4 = px.pie(values=[normal_conditions, warning_conditions], names=["Normal", "Warning"], title="Condition Distribution")
        st.plotly_chart(fig4)

        # Download button
        csv = predictions_df.to_csv(index=False)
        st.download_button("Download Complete History", data=csv, file_name="fuel_predictions_history.csv", mime="text/csv")
    else:
        st.write("No predictions have been made yet.")
