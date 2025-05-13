import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from shapely.geometry import Polygon
import alphashape

# Load CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'field_id' not in df.columns or 'lat' not in df.columns or 'lng' not in df.columns:
        st.error("CSV must contain 'field_id', 'lat', and 'lng' columns.")
        st.stop()

    # Sidebar filter
    st.sidebar.header("Filters")
    selected_fields = st.sidebar.multiselect(
        "Select field_id(s):",
        options=df['field_id'].unique(),
        default=df['field_id'].unique()
    )

    # Filtered data
    fields = df[df['field_id'].isin(selected_fields)]
    valid_fields = fields['field_id'].unique()

    # Show options
    show_points = st.sidebar.checkbox("Show Points", value=True)
    show_hull_points = st.sidebar.checkbox("Show Hull (Concave)", value=True)

    # Base map
    if not fields.empty:
        map_center = [fields['lat'].mean(), fields['lng'].mean()]
    else:
        map_center = [19.0, 72.0]

    m = folium.Map(location=map_center, zoom_start=12)

    def generate_more_hull_points(hull_points):
        """Linear interpolation between hull points to generate smoother polygon edges."""
        more_pts = []
        for i in range(len(hull_points)):
            pt1 = hull_points[i]
            pt2 = hull_points[(i + 1) % len(hull_points)]
            steps = 10
            for j in range(steps):
                lat = pt1[0] + (pt2[0] - pt1[0]) * j / steps
                lng = pt1[1] + (pt2[1] - pt1[1]) * j / steps
                more_pts.append([lat, lng])
        return np.array(more_pts)

    # Show original points
    if show_points:
        for field_id in valid_fields:
            points = fields[fields['field_id'] == field_id][['lat', 'lng']].values
            for lat, lng in points:
                folium.CircleMarker(
                    location=[lat, lng],
                    radius=3,
                    color='blue',
                    fill=True,
                    fill_opacity=0.6,
                    popup=str(field_id)
                ).add_to(m)

    # Show concave hull
    if show_hull_points:
        for field_id in valid_fields:
            points = fields[fields['field_id'] == field_id][['lat', 'lng']].values
            if len(points) < 4:
                continue
            try:
                alpha_shape = alphashape.alphashape(points, 0.01)
                if alpha_shape.geom_type == 'Polygon':
                    hull_pts = np.array(alpha_shape.exterior.coords)
                    folium.Polygon(locations=hull_pts.tolist(), color='green', fill=True, fill_opacity=0.5).add_to(m)
                    more_pts = generate_more_hull_points(hull_pts)
                    folium.PolyLine(locations=more_pts.tolist(), color='yellow', weight=2).add_to(m)
            except Exception as e:
                print(f"Failed to compute hull for field {field_id}: {e}")

    # Render map
    folium_static(m)
