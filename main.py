import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = "./data"

st.set_page_config(layout="wide", 
                   page_title="Climate & Precip Dashboard")

@st.cache_data
def load_data():
    in_path = f"{DATA_PATH}/processed/joined/joined_dataset.csv"
    df = pd.read_csv(in_path)
    
    co2_map = (df[['YEAR', 'ANNUAL_EMISSIONS_CHILE']]
               .drop_duplicates()
               .set_index('YEAR')
               )
    co2_map = co2_map['ANNUAL_EMISSIONS_CHILE'].to_dict()
    
    df['LAGGED_CHILE_CO2'] = (df['YEAR']
                              .apply(lambda y: co2_map.get(y - 1, np.nan)))
    
    annual_df = df.groupby(['REGION', 'YEAR']).agg({
        'MM_TOTAL_PRECIP': 'sum',
        'GEOGRAPHIC_ZONE': 'first',
        'LAGGED_CHILE_CO2': 'first',
        'LATITUDE': 'first'
    }).reset_index()
    
    annual_df = annual_df.rename({"YEAR": "Year", 
                                  "REGION": "Region",
                                  "MM_TOTAL_PRECIP": "Precipitation_mm",
                                  "GEOGRAPHIC_ZONE": "Geographic_zone",
                                  "LAGGED_CHILE_CO2": "Chile_CO2_ppm",
                                  "LATITUDE": "Latitude"
                                  }, axis=1)
    
    annual_df = annual_df.sort_values(by=["Region", "Year"])
    
    monthly_df = df.groupby(["REGION", 'YEAR', "MONTH"]).agg({
        'MM_TOTAL_PRECIP': 'sum',
        'GEOGRAPHIC_ZONE': 'first',
        'LAGGED_CHILE_CO2': 'first',
        'LATITUDE': 'first'
    }).reset_index()
    
    monthly_df = monthly_df.rename({"YEAR": "Year", 
                                  "MONTH": "Month",
                                  "REGION": "Region",
                                  "MM_TOTAL_PRECIP": "Precipitation_mm",
                                  "GEOGRAPHIC_ZONE": "Geographic_zone",
                                  "LAGGED_CHILE_CO2": "Chile_CO2_ppm",
                                  "LATITUDE": "Latitude"
                                  }, axis=1)
    
    annual_df = annual_df.sort_values(by=["Region", "Year"])
    monthly_df = monthly_df.sort_values(by=["Region", "Year", "Month"])
    
    return annual_df, monthly_df

annual_df, monthly_df = load_data()

with st.sidebar:
    st.header("Configuration")
    
    selected_region = st.selectbox("Select Region:", 
                                   annual_df["Region"].unique())
    
    st.markdown("---")
    st.markdown("""
    **Dashboard Guide:**
    - **Top Left:** Rainfall trends (1980-2024).
    - **Top Right:** Chilean CO₂ emissions context.
    - **Bottom Left:** Is Higher CO₂ correlating with less rain?
    - **Bottom Right:** Wet/dry months over time.
    """)

region_data = annual_df[annual_df["Region"] == selected_region]
region_monthly_df = monthly_df[monthly_df["Region"] == selected_region]

annual_data = (region_data
               .groupby("Year")[["Precipitation_mm", "Chile_CO2_ppm"]]
               .mean()
               .reset_index()
               )

st.title("Precipitation & Climate Analysis")
st.subheader(f"Region {selected_region}")

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)


with row1_col1:
    st.subheader("Annual Precipitation Trend (with Trendline)")
    fig1 = px.scatter(
        annual_data, 
        x="Year", 
        y="Precipitation_mm", 
        trendline="ols",
        trendline_color_override="red",
        title="Avg Monthly Rainfall per Year"
    )
    fig1.data[0].mode = 'lines+markers'
    fig1.update_layout(height=320, 
                       margin=dict(l=20, r=20, t=40, b=20))
    fig1.data[0].name = "Observed Rain"
    fig1.data[0].showlegend = True
    if len(fig1.data) > 1:
        fig1.data[1].name = "Trend (OLS)"
        fig1.data[1].showlegend = True
    st.plotly_chart(fig1, width='stretch')


with row1_col2:
    st.subheader("Chile CO₂ Emissions Context")
    global_co2 = (annual_df
                  .groupby("Year")["Chile_CO2_ppm"]
                  .mean()
                  .reset_index()
                  )
    
    fig3 = px.area(
        global_co2,
        x="Year",
        y="Chile_CO2_ppm",
        title="Chile CO₂ Concentration (ppm) 1980-2024",
        color_discrete_sequence=["#555555"] # Grey/Black for Carbon
    )
    fig3.update_traces(showlegend=True, name="Emissions")
    fig3.update_layout(height=320, 
                       margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig3, width='stretch')


with row2_col1:
    st.subheader(f"Correlation: CO₂ vs. Rainfall in {selected_region}")
    fig2 = px.scatter(
        annual_data,
        x="Chile_CO2_ppm",
        y="Precipitation_mm",
        color="Year",
        color_continuous_scale="Viridis",
        title="Does higher CO₂ imply more rain?",
        labels={"Chile_CO2_ppm": "Chile CO₂ (ppm)", 
                "Precipitation_mm": "Avg Precip (mm)"}
    )
    fig2.update_layout(height=320, 
                       margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig2, width='stretch')


with row2_col2:
    st.subheader("Precipitation Heatmap (Seasonality)")
    
    heatmap_data = region_monthly_df.pivot(index="Month", 
                                           columns="Year", 
                                           values="Precipitation_mm")
    
    fig4 = px.imshow(
        heatmap_data,
        labels=dict(x="Year", 
                    y="Month", 
                    color="Rain (mm)"),
        y=['Jan', 'Feb', 'Mar', 
           'Apr', 'May', 'Jun', 
           'Jul', 'Aug', 'Sep', 
           'Oct', 'Nov', 'Dec'],
        color_continuous_scale="RdBu",
        aspect="auto",
        origin='upper'
    )
    fig4.update_layout(height=320, 
                       margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig4, width='stretch')