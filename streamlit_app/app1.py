import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import os
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import seaborn as sns
from streamlit_autorefresh import st_autorefresh

# --- CONFIG ---
DATA_PATH = os.path.join("..", "data", "processed", "cleaned_data.csv")
RF_MODEL_PATH = os.path.join("..", "models", "RandomForest_model.joblib")

# --- REAL-TIME AUTO-REFRESH ---
st.set_page_config(page_title="Real-Time IndoFloods Dashboard", layout="wide")
refresh_interval = st.sidebar.slider("Auto-refresh interval (seconds)", 5, 60, 15)
st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")

# --- LOAD DATA (simulate real-time by adding random fluctuation) ---
@st.cache_data(ttl=refresh_interval)
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Simulate current river levels (replace with API in production)
    df['Current_Level'] = df['Danger Level'] * (0.85 + np.random.normal(0, 0.15, len(df)))
    return df

df = load_data()

# --- Helper: Risk Category ---
def get_risk_level(level):
    if pd.isna(level):
        return "Unknown"
    if level >= 300:
        return "Very High"
    elif level >= 200:
        return "High"
    elif level >= 100:
        return "Medium"
    elif level >= 50:
        return "Low"
    else:
        return "Very Low"

df["Risk_Level"] = df["Danger Level"].apply(get_risk_level)
df["Current_Risk"] = df["Current_Level"].apply(get_risk_level)

RISK_COLORS = {
    "Very High": "darkred",
    "High": "red",
    "Medium": "orange",
    "Low": "green",
    "Very Low": "blue",
    "Unknown": "gray"
}

# --- Session State for Monitoring ---
if "history" not in st.session_state:
    st.session_state["history"] = []
if "start_time" not in st.session_state:
    st.session_state["start_time"] = datetime.now()
if "filter_counts" not in st.session_state:
    st.session_state["filter_counts"] = {"State": {}, "Basin": {}, "Station": {}}

# --- TABS ---
tab_dashboard, tab_historical, tab_monitor = st.tabs(["🌊 Dashboard", "📈 Historical", "🖥️ Monitor"])

with tab_dashboard:
    # --- MAIN TITLE ---
    st.title("🌊 IndoFloods Real-Time Dashboard")
    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- SUMMARY STATS ---
    st.markdown("#### Dataset Summary")
    st.write(f"**Total Stations:** {len(df)}")
    st.write(f"**States:** {df['State'].nunique()}")
    st.write(f"**Basins:** {df['Basin'].nunique()}")
    st.write(f"**Date Range:** {df['Start_date'].min()} to {df['End_date'].max()}")

    # --- MOVED FILTERS TO MAIN AREA ---
    st.markdown("#### Filter Data")
    
    # Create a 3-column layout for filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 1. State dropdown
        states = ["All"] + sorted(df["State"].dropna().unique().tolist())
        state = st.selectbox("State", options=states)
    
    with col2:
        # 2. Basin dropdown, filtered by selected state
        if state != "All":
            basins = sorted(df[df["State"] == state]["Basin"].dropna().unique().tolist())
            basin_options = ["All"] + basins
        else:
            basin_options = ["All"] + sorted(df["Basin"].dropna().unique().tolist())
        basin = st.selectbox("Basin", options=basin_options)
    
    with col3:
        # 3. Station dropdown, filtered by selected state and basin
        if state != "All" and basin != "All":
            stations = sorted(df[(df["State"] == state) & (df["Basin"] == basin)]["Station"].dropna().unique().tolist())
            station_options = ["All"] + stations
        elif state != "All":
            stations = sorted(df[df["State"] == state]["Station"].dropna().unique().tolist())
            station_options = ["All"] + stations
        elif basin != "All":
            stations = sorted(df[df["Basin"] == basin]["Station"].dropna().unique().tolist())
            station_options = ["All"] + stations
        else:
            station_options = ["All"] + sorted(df["Station"].dropna().unique().tolist())
        station = st.selectbox("Station", options=station_options)

    # Log filter usage
    now = datetime.now()
    action = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "state": state,
        "basin": basin,
        "station": station
    }
    st.session_state["history"].append(action)
    for filter_name, value in [("State", state), ("Basin", basin), ("Station", station)]:
        if value != "All":
            st.session_state["filter_counts"][filter_name][value] = st.session_state["filter_counts"][filter_name].get(value, 0) + 1

    # --- FILTER DATA ---
    filtered_df = df.copy()
    if state != "All":
        filtered_df = filtered_df[filtered_df["State"] == state]
    if basin != "All":
        filtered_df = filtered_df[filtered_df["Basin"] == basin]
    if station != "All":
        filtered_df = filtered_df[filtered_df["Station"] == station]

    # --- DISPLAY CURRENT SELECTION ---
    if state != "All" or basin != "All" or station != "All":
        st.markdown("#### Current Selection")
        if state != "All":
            st.write(f"**Selected State:** {state}")
        if basin != "All":
            st.write(f"**Selected Basin:** {basin}")
        if station != "All":
            st.write(f"**Selected Station:** {station}")

    # --- DISPLAY TABLE ---
    with st.expander("Show Filtered Data Table"):
        st.dataframe(filtered_df)

    # --- FOLIUM MAP ---
    st.markdown("#### Interactive Flood Risk Map")
    filtered_df = filtered_df.dropna(subset=["Latitude", "Longitude"])
    if filtered_df.empty:
        st.warning("No valid coordinates found for the selected filters.")
    else:
        m = folium.Map(location=[filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean()], zoom_start=5)
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in filtered_df.iterrows():
            color = RISK_COLORS.get(row["Current_Risk"], "gray")
            popup_html = f"""
            <b>Station:</b> {row['Station']}<br>
            <b>GaugeID:</b> {row['GaugeID']}<br>
            <b>State:</b> {row['State']}<br>
            <b>Basin:</b> {row['Basin']}<br>
            <b>Current Level:</b> {row['Current_Level']:.2f}<br>
            <b>Danger Level:</b> {row['Danger Level']}<br>
            <b>Warning Level:</b> {row['Warning Level']}<br>
            <b>Risk Level:</b> <span style='color:{color};font-weight:bold'>{row['Current_Risk']}</span>
            """
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"{row['Station']} ({row['Current_Risk']} Risk)"
            ).add_to(marker_cluster)
        legend_html = """
        <div style="position: fixed; 
             bottom: 50px; left: 50px; width: 200px; height: 135px; 
             background-color: white; z-index:9999; font-size:14px;
             border:2px solid grey; border-radius:8px; padding: 10px;">
        <b>Risk Level Legend</b><br>
        <span style='color:darkred;'>●</span> Very High<br>
        <span style='color:red;'>●</span> High<br>
        <span style='color:orange;'>●</span> Medium<br>
        <span style='color:green;'>●</span> Low<br>
        <span style='color:blue;'>●</span> Very Low
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        st_folium(m, width=800, height=500)

    # --- EDA Plots ---
    st.markdown("#### EDA: Danger Level Distribution")
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.histplot(df["Danger Level"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # --- RandomForest Prediction Section ---
    st.markdown("#### Predict Danger Level for a Station (RandomForest Model)")
    @st.cache_resource
    def load_rf_model():
        if os.path.exists(RF_MODEL_PATH):
            return joblib.load(RF_MODEL_PATH)
        return None
    rf_model = load_rf_model()
    if rf_model is not None:
        if not filtered_df.empty:
            station_options = filtered_df["Station"].unique().tolist()
        else:
            station_options = df["Station"].unique().tolist()
        pred_station = st.selectbox("Select Station for Prediction", sorted(station_options))
        station_row = df[df["Station"] == pred_station].iloc[0]
        input_features = [
            "Warning Level", "Level_Entries", "Streamflow_Entries", "Source Catchment Area",
            "Catchment Area", "Area variation (%)"
        ]
        input_df = pd.DataFrame([station_row[input_features].values], columns=input_features)
        pred = rf_model.predict(input_df)[0]
        pred_risk = get_risk_level(pred)
        pred_color = RISK_COLORS.get(pred_risk, "gray")
        st.markdown(
            f"**Predicted Danger Level for {pred_station}:** "
            f"<span style='color:{pred_color};font-weight:bold'>{pred:.2f} ({pred_risk} Risk)</span>",
            unsafe_allow_html=True
        )
    else:
        st.warning("RandomForest model not found. Please train and save it as 'RandomForest_model.joblib' in the models directory.")

    now = datetime.now()
    elapsed = now - st.session_state["start_time"]
    elapsed_min = int(elapsed.total_seconds() // 60)
    st.info(f"**Current session duration:** {elapsed_min} minutes")

    st.markdown("---")
    st.caption("Flood Risk Dashboard | IndoFloods | Powered by Streamlit, Folium")

with tab_historical:
    st.header("📈 Historical Data Analysis")
    if not filtered_df.empty:
        selected_station = st.selectbox("Select Station for Historical Analysis", sorted(filtered_df["Station"].unique().tolist()), key="hist_station")
        station_data = df[df["Station"] == selected_station]
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        hist_levels = station_data['Danger Level'].values[0] * (0.7 + 0.6 * np.random.random(size=30))
        historical_df = pd.DataFrame({'Date': dates, 'Water_Level': hist_levels})
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(historical_df['Date'], historical_df['Water_Level'], marker='o', linestyle='-')
        ax.axhline(y=station_data['Danger Level'].values[0], color='r', linestyle='--', label='Danger Level')
        ax.axhline(y=station_data['Warning Level'].values[0], color='orange', linestyle='--', label='Warning Level')
        ax.fill_between(historical_df['Date'], historical_df['Water_Level'],
                        station_data['Danger Level'].values[0],
                        where=(historical_df['Water_Level'] > station_data['Danger Level'].values[0]),
                        color='red', alpha=0.3)
        ax.set_title(f'Historical Water Levels at {selected_station}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Water Level (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Maximum Level", f"{historical_df['Water_Level'].max():.2f} m")
        with col2:
            st.metric("Average Level", f"{historical_df['Water_Level'].mean():.2f} m")
        with col3:
            days_above_danger = (historical_df['Water_Level'] > station_data['Danger Level'].values[0]).sum()
            st.metric("Days Above Danger Level", f"{days_above_danger} days", f"{days_above_danger/30*100:.1f}%")
    else:
        st.info("Select a station to view historical analysis.")

with tab_monitor:
    st.header("🖥️ Monitor: User Activity & Analytics")
    now = datetime.now()
    elapsed = now - st.session_state["start_time"]
    elapsed_min = int(elapsed.total_seconds() // 60)
    st.write(f"**Session duration:** {elapsed_min} minutes")
    st.markdown("#### Filter/Search History")
    hist_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(hist_df.tail(50))
    st.markdown("#### Filter Usage Analytics")
    for filter_name, counts in st.session_state["filter_counts"].items():
        if counts:
            st.write(f"**{filter_name} usage:**")
            usage_df = pd.DataFrame(list(counts.items()), columns=[filter_name, "Count"]).sort_values("Count", ascending=False)
            st.dataframe(usage_df)
            fig1, ax1 = plt.subplots()
            ax1.pie(usage_df["Count"], labels=usage_df[filter_name], autopct='%1.1f%%')
            ax1.set_title(f"{filter_name} Selection Distribution")
            st.pyplot(fig1)
            fig2, ax2 = plt.subplots()
            sns.barplot(x=filter_name, y="Count", data=usage_df, ax=ax2)
            ax2.set_title(f"{filter_name} Usage Frequency")
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig2)
        else:
            st.write(f"No {filter_name} filters used yet.")
    if len(hist_df) > 0:
        st.markdown("#### Time-based Usage Analytics")
        if 'timestamp' in hist_df.columns:
            if hist_df['timestamp'].dtype == 'object':
                hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
            hist_df['hour'] = hist_df['timestamp'].dt.hour
            hour_counts = hist_df['hour'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=hour_counts.index, y=hour_counts.values, ax=ax)
            ax.set_title("Activity by Hour of Day")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Number of Actions")
            st.pyplot(fig)
    st.markdown("---")
    st.caption("Monitor tab: Tracks your filter/search history, session duration, and filter usage.")
    st.caption("Data tab: Displays the filtered data table and an interactive map of flood risk levels.")
    st.caption("Use the filters at the top to filter data by State, Basin, and Station.")
