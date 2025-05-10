import streamlit as st
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import folium
from folium import plugins
from geopy.distance import geodesic
from datetime import datetime

# Functions
def calculate_convex_hull_area(points):
    if len(points) < 3:
        return 0
    try:
        hull = ConvexHull(points)
        poly = Polygon(points[hull.vertices])
        return poly.area
    except Exception:
        return 0

def calculate_centroid(points):
    return np.mean(points, axis=0)

def generate_more_hull_points(points, num_splits=3):
    new_points = []
    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]
        new_points.append(start_point)
        for j in range(1, num_splits):
            intermediate_point = start_point + j * (end_point - start_point) / num_splits
            new_points.append(intermediate_point)
    return np.array(new_points)

def process_csv_data(gps_data, show_hull_points):
    gps_data['Timestamp'] = pd.to_datetime(gps_data['time'], unit='ms')
    gps_data['lat'] = gps_data['lat'].astype(float)
    gps_data['lng'] = gps_data['lon'].astype(float)

    coords = gps_data[['lat', 'lng']].values
    db = DBSCAN(eps=0.0000235, min_samples=20).fit(coords)
    gps_data['field_id'] = db.labels_

    fields = gps_data[gps_data['field_id'] != -1]
    field_areas = fields.groupby('field_id').apply(lambda df: calculate_convex_hull_area(df[['lat', 'lng']].values))
    field_areas_m2 = field_areas * 0.77 * (111000 ** 2)
    field_areas_gunthas = field_areas_m2 / 101.17

    field_times = fields.groupby('field_id').apply(
        lambda df: (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds() / 60.0
    )

    field_dates = fields.groupby('field_id').agg(
        start_date=('Timestamp', 'min'),
        end_date=('Timestamp', 'max')
    )

    valid_fields = field_areas_gunthas[field_areas_gunthas >= 5].index
    field_areas_gunthas = field_areas_gunthas[valid_fields]
    field_times = field_times[valid_fields]
    field_dates = field_dates.loc[valid_fields]

    centroids = fields.groupby('field_id').apply(lambda df: calculate_centroid(df[['lat', 'lng']].values))

    travel_distances = []
    travel_times = []
    field_ids = list(valid_fields)

    if len(field_ids) > 1:
        for i in range(len(field_ids) - 1):
            c1 = centroids.loc[field_ids[i]]
            c2 = centroids.loc[field_ids[i + 1]]
            distance = geodesic(c1, c2).kilometers
            time = (field_dates.loc[field_ids[i + 1], 'start_date'] - field_dates.loc[field_ids[i], 'end_date']).total_seconds() / 60.0
            travel_distances.append(distance)
            travel_times.append(time)

        for i in range(len(field_ids) - 1):
            end = fields[fields['field_id'] == field_ids[i]][['lat', 'lng']].values[-1]
            start = fields[fields['field_id'] == field_ids[i + 1]][['lat', 'lng']].values[0]
            dist = geodesic(end, start).kilometers
            time = (field_dates.loc[field_ids[i + 1], 'start_date'] - field_dates.loc[field_ids[i], 'end_date']).total_seconds() / 60.0
            travel_distances.append(dist)
            travel_times.append(time)

        travel_distances.append(np.nan)
        travel_times.append(np.nan)
    else:
        travel_distances.append(np.nan)
        travel_times.append(np.nan)

    if len(travel_distances) != len(field_areas_gunthas):
        travel_distances = travel_distances[:len(field_areas_gunthas)]
        travel_times = travel_times[:len(field_areas_gunthas)]

    combined_df = pd.DataFrame({
        'Field ID': field_areas_gunthas.index,
        'Area (Gunthas)': field_areas_gunthas.values,
        'Time (Minutes)': field_times.values,
        'Start Date': field_dates['start_date'].values,
        'End Date': field_dates['end_date'].values,
        'Travel Distance to Next Field (km)': travel_distances,
        'Travel Time to Next Field (minutes)': travel_times
    })

    total_area = field_areas_gunthas.sum()
    total_time = field_times.sum()
    total_travel_distance = np.nansum(travel_distances)
    total_travel_time = np.nansum(travel_times)

    map_center = [gps_data['lat'].mean(), gps_data['lng'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    folium.TileLayer(
        tiles='https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1IjoiZmxhc2hvcDAwNyIsImEiOiJjbHo5NzkycmIwN2RxMmtzZHZvNWpjYmQ2In0.A_FZYl5zKjwSZpJuP_MHiA',
        attr='Mapbox Satellite',
        name='Satellite',
        overlay=True,
        control=True
    ).add_to(m)
    plugins.Fullscreen().add_to(m)

    for _, row in gps_data.iterrows():
        color = 'blue' if row['field_id'] in valid_fields else 'red'
        folium.CircleMarker(location=[row['lat'], row['lng']], radius=2, color=color, fill=True).add_to(m)

    if show_hull_points:
        for field_id in valid_fields:
            points = fields[fields['field_id'] == field_id][['lat', 'lng']].values
            hull = ConvexHull(points)
            hull_pts = points[hull.vertices]
            folium.Polygon(locations=hull_pts.tolist(), color='green', fill=True, fill_opacity=0.5).add_to(m)
            more_pts = generate_more_hull_points(hull_pts)
            folium.PolyLine(locations=more_pts.tolist(), color='yellow', weight=2).add_to(m)

    return m, combined_df, total_area, total_time, total_travel_distance, total_travel_time

# Streamlit App
def main():
    st.title("Field CSV Analyzer with Area & Travel Metrics")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    show_hull_points = st.checkbox("Show Hull Points", value=False)

    if uploaded_file:
        gps_data = pd.read_csv(uploaded_file)

        required_columns = {'lat', 'lon', 'time'}
        if not required_columns.issubset(gps_data.columns):
            st.error("CSV must contain columns: lat, lon, time (time in ms).")
            return

        m, df, total_area, total_time, total_travel_dist, total_travel_time = process_csv_data(gps_data, show_hull_points)

        st.success("Analysis Completed.")
        st.subheader("Field Data Summary")
        st.dataframe(df)

        st.subheader("Total Metrics")
        st.markdown(f"**Total Area:** {total_area:.2f} gunthas")
        st.markdown(f"**Total Time:** {total_time:.2f} minutes")
        st.markdown(f"**Total Travel Distance:** {total_travel_dist:.2f} km")
        st.markdown(f"**Total Travel Time:** {total_travel_time:.2f} minutes")

        st.subheader("Field Map")
        folium_static(m)

# For folium map rendering in Streamlit
from streamlit_folium import folium_static

if __name__ == "__main__":
    main()
