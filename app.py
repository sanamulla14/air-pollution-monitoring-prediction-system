# Air Quality StreamLit Project

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import folium
from streamlit_folium import st_folium

# ---------------- PAGE SETUP ----------------

st.set_page_config(page_title="🌫 Air Quality Monitor", layout="wide")

st.title("🌍 Air Quality Dashboard with Prediction")

st.markdown(
    "Monitor, analyze and predict air pollution levels across Indian cities."
)

# ---------------- SIMULATED DATA ----------------

np.random.seed(42)

dates = pd.date_range(start='2026-01-01', end='2026-09-30')

cities = [
    'Delhi',
    'Mumbai',
    'Bengaluru',
    'Chennai',
    'Kolkata',
    'Hyderabad',
    'Ahmedabad'
]

all_data = []

for city in cities:

    for i in range(len(dates)):

        # Different AQI patterns for cities

        if city == 'Delhi':
            pm25 = np.random.normal(160, 25)
            pm10 = np.random.normal(220, 30)

        elif city == 'Mumbai':
            pm25 = np.random.normal(95, 20)
            pm10 = np.random.normal(140, 25)

        elif city == 'Bengaluru':
            pm25 = np.random.normal(55, 12)
            pm10 = np.random.normal(90, 15)

        elif city == 'Chennai':
            pm25 = np.random.normal(70, 15)
            pm10 = np.random.normal(110, 18)

        elif city == 'Kolkata':
            pm25 = np.random.normal(135, 20)
            pm10 = np.random.normal(190, 28)

        elif city == 'Hyderabad':
            pm25 = np.random.normal(85, 15)
            pm10 = np.random.normal(125, 20)

        else:  # Ahmedabad
            pm25 = np.random.normal(115, 18)
            pm10 = np.random.normal(170, 25)

        no2 = np.random.normal(40, 10)
        so2 = np.random.normal(20, 5)
        co2 = np.random.normal(400, 30)
        temp = np.random.normal(28, 4)
        humidity = np.random.normal(60, 10)
        wind = np.random.normal(3, 1)

        all_data.append([
            dates[i],
            city,
            pm25,
            pm10,
            no2,
            so2,
            co2,
            temp,
            humidity,
            wind
        ])

columns = [
    "Date",
    "City",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO2",
    "Temperature",
    "Humidity",
    "WindSpeed"
]

df = pd.DataFrame(all_data, columns=columns)

# AQI Formula

df['AQI'] = (
    0.4 * df['PM2.5'] +
    0.3 * df['PM10'] +
    0.15 * df['NO2'] +
    0.1 * df['SO2'] +
    0.05 * df['Humidity']
)

# ---------------- SIDEBAR ----------------

city = st.sidebar.selectbox(
    "Select City",
    df['City'].unique()
)

show_map = st.sidebar.checkbox(
    "🗺 Show City AQI Map",
    value=True
)

df_city = df[df['City'] == city]

# ---------------- MACHINE LEARNING ----------------

features = [
    'PM2.5',
    'PM10',
    'NO2',
    'SO2',
    'CO2',
    'Temperature',
    'Humidity',
    'WindSpeed'
]

X = df_city[features]

y = df_city['AQI']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------- MODEL PERFORMANCE ----------------

st.subheader(f"📊 AQI Prediction Performance for {city}")

st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

st.write(
    f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}"
)

# Small Summary

st.subheader("📌 Model Summary")

st.write(f"Total Records: {len(df_city)}")

st.write(f"Features Used: {len(features)}")

st.write("Model Used: Linear Regression")

# ---------------- AQI CATEGORY ----------------

def get_aqi_category_color(aqi_value):

    if aqi_value <= 50:
        return "🟢 Good", "#66bb6a"

    elif aqi_value <= 100:
        return "🟡 Satisfactory", "#ffee58"

    elif aqi_value <= 200:
        return "🟠 Moderate", "#ffa726"

    elif aqi_value <= 300:
        return "🔴 Poor", "#ef5350"

    elif aqi_value <= 400:
        return "🟣 Very Poor", "#8e24aa"

    else:
        return "🟤 Severe", "#6d4c41"

# ---------------- HEALTH ADVISORY ----------------

def get_health_advisory(aqi):

    if aqi <= 50:
        return (
            "🟢 Good",
            "Air quality is excellent. Perfect for outdoor activities 🌿",
            "✅ Morning walk recommended\n✅ Open windows for fresh air"
        )

    elif aqi <= 100:
        return (
            "🟡 Satisfactory",
            "Air quality is acceptable for most people.",
            "😷 Sensitive people should wear masks\n💧 Stay hydrated"
        )

    elif aqi <= 200:
        return (
            "🟠 Moderate",
            "Breathing discomfort possible for sensitive groups.",
            "😷 Wear a mask outdoors\n🥤 Drink more water\n🧴 Apply sunscreen during daytime"
        )

    elif aqi <= 300:
        return (
            "🔴 Poor",
            "Air pollution may cause health discomfort.",
            "😷 Avoid jogging outside\n🚪 Keep windows closed\n🌳 Avoid high traffic areas"
        )

    elif aqi <= 400:
        return (
            "🟣 Very Poor",
            "Health warnings issued for everyone.",
            "😷 Use N95 masks\n🏠 Stay indoors\n💨 Use air purifiers if possible"
        )

    else:
        return (
            "⚫ Severe",
            "Hazardous air quality. Serious health risks possible.",
            "🚫 Avoid outdoor activities\n😷 Use protective masks\n🏥 Seek medical help if breathing issues occur"
        )

# ---------------- CURRENT AQI ----------------

latest_aqi = df_city.sort_values(
    'Date',
    ascending=False
).iloc[0]['AQI']

category, color = get_aqi_category_color(latest_aqi)

st.markdown(
    f"""
    <div style='background-color:{color};
    padding:12px;
    border-radius:8px'>

    <h3 style='text-align:center;color:black'>

    Current AQI in <b>{city}</b>: 
    {latest_aqi:.2f} – {category}

    </h3>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- HEALTH DISPLAY ----------------

level, advisory, precautions = get_health_advisory(
    latest_aqi
)

st.markdown(f"### 🧑‍⚕ Health Advisory: {level}")

st.warning(advisory)

st.markdown("### ⚠ Recommended Precautions")

st.success(precautions)

# ---------------- AQI TREND ----------------

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 📈 AQI Over Time")

    fig1, ax1 = plt.subplots(figsize=(6,4))

    sns.lineplot(
        data=df_city,
        x='Date',
        y='AQI',
        linewidth=2,
        color='darkred',
        ax=ax1
    )

    ax1.set_title("Daily AQI")

    ax1.tick_params(axis='x', rotation=45)

    st.pyplot(fig1)

# ---------------- HEATMAP ----------------

with col2:

    st.markdown("### 🧪 Correlation Heatmap")

    fig2, ax2 = plt.subplots(figsize=(6,5))

    sns.heatmap(
        df_city[features + ['AQI']].corr(),
        annot=True,
        cmap='coolwarm',
        ax=ax2
    )

    ax2.set_title("Feature Correlation")

    st.pyplot(fig2)

# ---------------- POLLUTION DISTRIBUTION ----------------

st.markdown("### 📦 Pollution Level Distribution")

df_melted = df_city.melt(
    id_vars=['Date'],
    value_vars=['PM2.5', 'PM10', 'NO2', 'SO2']
)

fig3, ax3 = plt.subplots(figsize=(8,4))

sns.boxplot(
    x='variable',
    y='value',
    data=df_melted,
    ax=ax3
)

ax3.set_title("Pollution Distribution")

st.pyplot(fig3)

# ---------------- TEMPERATURE VS AQI ----------------

col3, col4 = st.columns(2)

with col3:

    st.markdown("### 🌡 Temperature vs AQI")

    fig4, ax4 = plt.subplots(figsize=(6,4))

    sns.scatterplot(
        data=df_city,
        x='Temperature',
        y='AQI',
        ax=ax4
    )

    ax4.set_title("Temperature vs AQI")

    st.pyplot(fig4)

# ---------------- ACTUAL VS PREDICTED ----------------

with col4:

    st.markdown("### 🔮 Predicted vs Actual AQI")

    fig5, ax5 = plt.subplots(figsize=(6,4))

    sns.scatterplot(
        x=y_test,
        y=y_pred,
        ax=ax5
    )

    ax5.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--'
    )

    ax5.set_xlabel("Actual AQI")

    ax5.set_ylabel("Predicted AQI")

    ax5.set_title("Prediction Accuracy")

    st.pyplot(fig5)

# ---------------- MONTHLY ANALYSIS ----------------

df['Month'] = df['Date'].dt.month_name()

df['Month_Num'] = df['Date'].dt.month

with st.expander("📈 Seasonal Variation Analysis"):

    st.markdown("### Monthly AQI Trend")

    df_monthly = df[df['City'] == city].groupby(
        ['Month', 'Month_Num']
    )['AQI'].mean().reset_index()

    df_monthly = df_monthly.sort_values(
        'Month_Num'
    )

    fig6, ax6 = plt.subplots(figsize=(8, 4))

    sns.barplot(
        x='Month',
        y='AQI',
        data=df_monthly,
        palette='coolwarm',
        ax=ax6
    )

    ax6.set_title("Average AQI by Month")

    st.pyplot(fig6)

# ---------------- SIMPLE AQI PREDICTION ----------------

st.markdown("## 🔮 Predict AQI")

pm25 = st.number_input(
    "Enter PM2.5",
    value=80.0
)

pm10 = st.number_input(
    "Enter PM10",
    value=120.0
)

no2 = st.number_input(
    "Enter NO2",
    value=40.0
)

so2 = st.number_input(
    "Enter SO2",
    value=20.0
)

if st.button("Predict AQI"):

    predicted_aqi = (
        0.4 * pm25 +
        0.3 * pm10 +
        0.15 * no2 +
        0.1 * so2
    )

    pred_category, pred_color = get_aqi_category_color(
        predicted_aqi
    )

    st.success(
        f"Predicted AQI: {predicted_aqi:.2f} – {pred_category}"
    )

# ---------------- DOWNLOAD OPTION ----------------

st.markdown("### 📥 Download AQI Data")

csv = df_city.to_csv(index=False).encode('utf-8')

st.download_button(
    "Download City AQI Data",
    csv,
    f"{city}_AQI.csv",
    "text/csv"
)

# ---------------- MAP ----------------

city_coords = {

    'Delhi': (28.6139, 77.2090),

    'Mumbai': (19.0760, 72.8777),

    'Bengaluru': (12.9716, 77.5946),

    'Chennai': (13.0827, 80.2707),

    'Kolkata': (22.5726, 88.3639),

    'Ahmedabad': (23.0225, 72.5714),

    'Hyderabad': (17.3850, 78.4867)
}

def get_aqi_color(aqi):

    if aqi <= 50:
        return 'green'

    elif aqi <= 100:
        return 'yellow'

    elif aqi <= 200:
        return 'orange'

    elif aqi <= 300:
        return 'red'

    elif aqi <= 400:
        return 'purple'

    else:
        return 'maroon'

if show_map:

    st.markdown(
        "### 🗺 Air Quality Map of Indian Cities"
    )

    latest_aqi_data = df.groupby('City').apply(
        lambda x: x.loc[x['Date'].idxmax()]
    ).reset_index(drop=True)

    m = folium.Map(
        location=[22.0, 80.0],
        zoom_start=5
    )

    for _, row in latest_aqi_data.iterrows():

        city_name = row['City']

        lat, lon = city_coords[city_name]

        aqi = row['AQI']

        color = get_aqi_color(aqi)

        folium.CircleMarker(
            location=(lat, lon),
            radius=10 + (aqi / 50),
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"🌍 {city_name}<br>"
                f"AQI: {aqi:.1f}<br>"
                f"Status: {category}",
                max_width=200
            )
        ).add_to(m)

    st_folium(m, width=700, height=500)