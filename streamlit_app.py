import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib, pickle

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

# Template Configuration
st.markdown(dark, unsafe_allow_html=True)

# Streamlit app
st.subheader("Mobile Price Classification")

# Features
# Bluetooh
blue = st.checkbox("Choose whether the phone has bluetooth or not.", value=True)
if blue: st.write("The phone has bluetooth!")

# 3G
three_g = st.checkbox("Choose whether the phone has 3G or not.", value=True)
if three_g: st.write("The phone has 3G!")

# 4G
four_g = st.checkbox("Choose whether the phone has 4G or not.", value=True)
if four_g: st.write("The phone has 4G!")

# Touch Screen
touch_screen = st.checkbox("Choose whether the phone has touch screen or not.", value=True)
if touch_screen: st.write("The phone has touch screen!")

# Wifi
wifi = st.checkbox("Choose whether the phone has wifi or not.", value=True)
if wifi: st.write("The phone has wifi!")

# Dual SIM 
dual_sim = st.checkbox("Choose whether the phone has dual sim support or not.", value= False)  
if dual_sim: st.write("The phone supports dual sim.")

# Battery_power
battery_power = st.slider('Total energy a battery can store in one time measured in mAh.', min_value=int(501), max_value=int(1998), step=1, value=1238)

# Clock Speed
clock_speed = st.slider('speed at which microprocessor executes instructions.', min_value=0.5, max_value=3.0, value=1.52)

# Front Camera mega pixels
fc = st.slider('Front Camera mega pixels.', min_value=int(0), max_value=int(19), step=1, value=5)

# Internal Memory in Gigabytes.
int_memory = st.slider('Internal Memory in Gigabytes.', min_value=int(2), max_value=int(64), step=1, value=32)

# Mobile Depth in cm.
m_dep = st.slider('Mobile Depth in cm.', min_value=0.1, max_value=1.0, value=0.5)

#  Weight of mobile phone.
mobile_wt = st.slider('Weight of mobile phone.', min_value=int(80), max_value=int(200), step=1, value=140)

# Number of cores of processor.
n_cores = st.slider('Number of cores of processor.', min_value=int(1), max_value=int(8), step=1, value=4)




classes = ['Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost']
probabilities = np.random.rand(len(classes))

if st.button("Submit"):

    st.write("The following figure shows arbitrary probabilities for now. We will get the predictions from a ML model soon. Stay tuned!")

    # Bar chart showing probabilities for each class
    df_prob = pd.DataFrame({'Class': classes, 'Probability': probabilities})
    chart = alt.Chart(df_prob).mark_bar().encode(
        x='Probability',
        y=alt.Y('Class', sort='-x')
    ).properties(
        width=500,
        height=200
    )
    st.altair_chart(chart, use_container_width=True)


