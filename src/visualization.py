import matplotlib.pyplot as plt
import seaborn as sns
import shap
import folium
from folium.plugins import MarkerCluster

def plot_actual_vs_pred(y_test, y_pred, model_name):
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.show()

def plot_residuals(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 3))
    sns.histplot(residuals, kde=True)
    plt.title(f"{model_name} - Residuals Distribution")
    plt.show()

def plot_shap_summary(model, X_train, X_test, model_name):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Summary Plot - {model_name}")
    plt.show()

def create_risk_map(df, lat_col="Latitude", lon_col="Longitude", risk_col="Risk_Level", popup_cols=None, save_path=None):
    """
    Create a folium map with points colored by risk level.
    """
    risk_colors = {'Very High':'darkred','High':'red','Medium':'orange','Low':'green','Very Low':'blue'}
    m = folium.Map(location=[df[lat_col].mean(), df[lon_col].mean()], zoom_start=5)
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in df.iterrows():
        color = risk_colors.get(row[risk_col], 'gray')
        popup_text = "<br>".join([f"<b>{col}:</b> {row[col]}" for col in (popup_cols or df.columns)])
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_text, max_width=350),
            tooltip=f"{row['Station']} ({row[risk_col]} Risk)"
        ).add_to(marker_cluster)
    if save_path:
        m.save(save_path)
    return m
