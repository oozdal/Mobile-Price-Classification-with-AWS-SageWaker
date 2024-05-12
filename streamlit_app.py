import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pickle

light = '''
<style>
    .stApp {
    background-color: white;
    }
</style>
'''

dark = '''
<style>
    .stApp {
    background-color: black;
    }
</style>
'''

def map_number_to_class(number):
    class_mapping = {
        0: 'Low Cost',
        1: 'Medium Cost',
        2: 'High Cost',
        3: 'Very High Cost'
    }
    return class_mapping.get(number, 'Unknown')

# Template Configuration
st.markdown(dark, unsafe_allow_html=True)

# Streamlit app
st.subheader("Mobile Price Classification")

# Features
# Bluetooh
blue = st.toggle("Choose whether the phone has bluetooth or not.", value=True)
if blue: st.write("The phone has bluetooth!")

# 3G
three_g = st.toggle("Choose whether the phone has 3G or not.", value=True)
if three_g: st.write("The phone has 3G!")

# 4G
four_g = st.toggle("Choose whether the phone has 4G or not.", value=True)
if four_g: st.write("The phone has 4G!")

# Touch Screen
touch_screen = st.toggle("Choose whether the phone has touch screen or not.", value=True)
if touch_screen: st.write("The phone has touch screen!")

# Wifi
wifi = st.toggle("Choose whether the phone has wifi or not.", value=True)
if wifi: st.write("The phone has wifi!")

# Dual SIM 
dual_sim = st.toggle("Choose whether the phone has dual sim support or not.", value= False)  
if dual_sim: st.write("The phone supports dual sim.")

# Battery_power
battery_power = st.slider('Total energy a battery can store in one time measured in mAh.', min_value=int(501), max_value=int(1998), step=1, value=1238)

# Clock Speed
clock_speed = st.slider('speed at which microprocessor executes instructions.', min_value=0.5, max_value=3.0, value=1.52)

# Front Camera mega pixels
fc = st.slider('Front Camera mega pixels.', min_value=int(0), max_value=int(19), step=1, value=5)

# int_memory: Internal Memory in Gigabytes.
int_memory = st.slider('Internal Memory in Gigabytes.', min_value=int(2), max_value=int(64), step=1, value=32)

# m_dep: Mobile Depth in cm.
m_dep = st.slider('Mobile Depth in cm.', min_value=0.1, max_value=1.0, value=0.5)

# mobile_wt: Weight of mobile phone.
mobile_wt = st.slider('Weight of mobile phone.', min_value=int(80), max_value=int(200), step=1, value=140)

# n_cores: Number of cores of processor.
n_cores = st.slider('Number of cores of processor.', min_value=int(1), max_value=int(8), step=1, value=4)

# pc: Primary Camera mega pixels.
pc = st.slider('Primary Camera mega pixels.', min_value=int(0), max_value=int(20), step=1, value=9)

# Pixel Resolution Height.
px_height = st.slider('Pixel Resolution Height.', min_value=int(1), max_value=int(1960), step=1, value=645)

# Pixel Resolution Width.
px_width = st.slider('Pixel Resolution Width.', min_value=int(500), max_value=int(1998), step=1, value=1251)

# Ram: Random Access Memory in Mega Bytes.
ram = st.slider('Random Access Memory in Mega Bytes.', min_value=int(256), max_value=int(3998), step=1, value=2124)

# sc_h: Screen Height of mobile in cm.
sc_h = st.slider('Screen Height of mobile in cm.', min_value=int(5), max_value=int(19), step=1, value=12)

#sc_w: Screen Width of mobile in cm.
sc_w = st.slider('Screen Width of mobile in cm.', min_value=int(0), max_value=int(18), step=1, value=5)

#talk_time: longest time that a single battery charge will last when you are.
talk_time = st.slider('Longest time that a single battery charge will last when you are.', 
            min_value=int(2), max_value=int(20), step=1, value=5)

if st.button("Submit"):

    feature_values = [
        battery_power,
        int(blue),
        clock_speed,
        int(dual_sim),
        fc,
        int(four_g),
        int_memory,
        m_dep,
        mobile_wt,
        n_cores,
        pc,
        px_height,
        px_width,
        ram,
        sc_h,
        sc_w,
        talk_time,
        int(three_g),
        int(touch_screen),
        int(wifi)]

    feature_names= ["battery_power",
        "blue",
        "clock_speed",
        "dual_sim",
        "fc",
        "four_g",
        "int_memory",
        "m_dep",
        "mobile_wt",
        "n_cores",
        "pc",
        "px_height",
        "px_width",
        "ram",
        "sc_h",
        "sc_w",
        "talk_time",
        "three_g",
        "touch_screen",
        "wifi"]

    # Return user-defined feature values
    feature_values_df = pd.DataFrame(np.array(feature_values).reshape(1, -1), columns=feature_names)    
    st.dataframe(feature_values_df)

    classes = ['Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost']

    # Load the model from disk
    filename = 'outputs/RandomForest.sav'
    rf_best_estimator = pickle.load(open(filename, 'rb'))

    # Predictions by Random Forest 
    y_pred_rf = rf_best_estimator.predict(feature_values_df)
    probabilities = rf_best_estimator.predict_proba(feature_values_df)
    max_prob = np.max(rf_best_estimator.predict_proba(feature_values_df), axis=1)[0] * 100

    # Returns the result
    st.success(f"This is a {map_number_to_class(y_pred_rf[0])} phone with a probability of {max_prob:.2f}%")

    # Bar chart showing probabilities for each class
    df_prob = pd.DataFrame({'Class': classes, 'Probability': probabilities[0]})
    chart = alt.Chart(df_prob).mark_bar().encode(
        x='Probability',
        y=alt.Y('Class', sort='-x')
    ).properties(
        width=500,
        height=200
    )
    st.altair_chart(chart, use_container_width=True)

