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

# --- CONFIG ---
DATA_PATH = os.path.join("..", "data", "processed", "cleaned_data.csv")
RF_MODEL_PATH = os.path.join("..", "models", "RandomForest_model.joblib")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# --- Helper: Risk Category ---
def get_risk_level(danger):
    if pd.isna(danger):
        return "Unknown"
    if danger >= 300:
        return "Very High"
    elif danger >= 200:
        return "High"
    elif danger >= 100:
        return "Medium"
    elif danger >= 50:
        return "Low"
    else:
        return "Very Low"

df["Risk_Level"] = df["Danger Level"].apply(get_risk_level)

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
tab_dashboard, tab_monitor = st.tabs(["🌊 Dashboard", "🖥️ Monitor"])

with tab_dashboard:
    # --- SIDEBAR FILTERS WITH DEPENDENT DROPDOWNS ---
    st.sidebar.header("Filter Data")
    
    # 1. State dropdown
    states = ["All"] + sorted(df["State"].dropna().unique().tolist())
    state = st.sidebar.selectbox("State", options=states)
    
    # 2. Basin dropdown, filtered by selected state
    if state != "All":
        basins = sorted(df[df["State"] == state]["Basin"].dropna().unique().tolist())
        basin_options = ["All"] + basins
    else:
        basin_options = ["All"] + sorted(df["Basin"].dropna().unique().tolist())
    
    basin = st.sidebar.selectbox("Basin", options=basin_options)
    
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
    
    station = st.sidebar.selectbox("Station", options=station_options)

    # Log filter usage
    now = datetime.now()
    action = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "state": state,
        "basin": basin,
        "station": station
    }
    st.session_state["history"].append(action)
    
    # Count filter usage
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

    # --- MAIN TITLE ---
    st.title("🌊 IndoFloods Risk Dashboard")

    # --- SUMMARY STATS ---
    st.markdown("#### Dataset Summary")
    st.write(f"**Total Stations:** {len(df)}")
    st.write(f"**States:** {df['State'].nunique()}")
    st.write(f"**Basins:** {df['Basin'].nunique()}")
    st.write(f"**Date Range:** {df['Start_date'].min()} to {df['End_date'].max()}")
    
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
    # Remove rows with NaN lat/lon
    filtered_df = filtered_df.dropna(subset=["Latitude", "Longitude"])
    if filtered_df.empty:
        st.warning("No valid coordinates found for the selected filters.")
    else:
        m = folium.Map(location=[filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean()], zoom_start=5)
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in filtered_df.iterrows():
            color = RISK_COLORS.get(row["Risk_Level"], "gray")
            popup_html = f"""
            <b>Station:</b> {row['Station']}<br>
            <b>GaugeID:</b> {row['GaugeID']}<br>
            <b>State:</b> {row['State']}<br>
            <b>Basin:</b> {row['Basin']}<br>
            <b>Warning Level:</b> {row['Warning Level']}<br>
            <b>Danger Level:</b> {row['Danger Level']}<br>
            <b>Catchment Area:</b> {row['Catchment Area']}<br>
            <b>Area variation (%):</b> {row['Area variation (%)']}<br>
            <b>Reliability:</b> {row['Reliability']}<br>
            <b>Risk Level:</b> <span style='color:{color};font-weight:bold'>{row['Risk_Level']}</span>
            """
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"{row['Station']} ({row['Risk_Level']} Risk)"
            ).add_to(marker_cluster)
        # Add legend
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
        st.success(f"Predicted Danger Level for {pred_station}: **{pred:.2f}**")
    else:
        st.warning("RandomForest model not found. Please train and save it as 'RandomForest_model.joblib' in the models directory.")

    # --- Session Duration in Dashboard ---
    now = datetime.now()
    elapsed = now - st.session_state["start_time"]
    elapsed_min = int(elapsed.total_seconds() // 60)
    st.info(f"**Current session duration:** {elapsed_min} minutes")

    st.markdown("---")
    st.caption("Flood Risk Dashboard | IndoFloods | Powered by Streamlit, Folium")

with tab_monitor:
    st.header("🖥️ Monitor: User Activity & Analytics")

    # --- Session Duration ---
    now = datetime.now()
    elapsed = now - st.session_state["start_time"]
    elapsed_min = int(elapsed.total_seconds() // 60)
    st.write(f"**Session duration:** {elapsed_min} minutes")

    # --- History Table ---
    st.markdown("#### Filter/Search History")
    hist_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(hist_df.tail(50))  # Show last 50 actions

    # --- Filter Usage Analytics with Visualizations ---
    st.markdown("#### Filter Usage Analytics")
    
    # Create columns for tables and charts
    for filter_name, counts in st.session_state["filter_counts"].items():
        if counts:
            st.write(f"**{filter_name} usage:**")
            
            # Create a DataFrame for this filter's usage
            usage_df = pd.DataFrame(list(counts.items()), columns=[filter_name, "Count"]).sort_values("Count", ascending=False)
            
            # Display in tabular format
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(usage_df)
            
            with col2:
                # Create pie chart
                if len(usage_df) > 0:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(usage_df["Count"], labels=usage_df[filter_name], autopct='%1.1f%%')
                    ax.set_title(f"{filter_name} Selection Distribution")
                    st.pyplot(fig)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=filter_name, y="Count", data=usage_df, ax=ax)
                    ax.set_title(f"{filter_name} Usage Frequency")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                    st.pyplot(fig)
        else:
            st.write(f"No {filter_name} filters used yet.")
    
    # --- Time-based Analytics ---
    if len(hist_df) > 0:
        st.markdown("#### Time-based Usage Analytics")
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in hist_df.columns:
            if hist_df['timestamp'].dtype == 'object':
                hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
            
            # Extract hour of day
            hist_df['hour'] = hist_df['timestamp'].dt.hour
            
            # Count actions by hour
            hour_counts = hist_df['hour'].value_counts().sort_index()
            
            # Plot hourly activity
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=hour_counts.index, y=hour_counts.values, ax=ax)
            ax.set_title("Activity by Hour of Day")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Number of Actions")
            st.pyplot(fig)

    st.markdown("---")
    st.caption("Monitor tab: Tracks your filter/search history, session duration, and filter usage.")
    st.caption("Data tab: Displays the filtered data table and an interactive map of flood risk levels.")
    st.caption("Use the sidebar to filter data by State, Basin, and Station.")
