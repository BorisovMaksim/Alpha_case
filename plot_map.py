import pandas as pd
import shapely.wkt
import folium
import numpy as np


def plot_map():
    gexons = pd.read_csv("C:/Users/maks/PycharmProjects/Alpha/final_table.csv")
    df_isochrones = pd.read_csv(r"C:/Users/maks/Documents/Alpha_case/train/isochrones_walk_dataset.csv")
    geo_data = df_isochrones.set_index('geo_h3_10').join(gexons.set_index('geo_h3_10'), how='right')[
        'walk_15min'].apply(
        lambda s: shapely.wkt.loads(s))
    cat = df_isochrones.set_index('geo_h3_10').join(gexons.set_index('geo_h3_10'), how='right').atm_cat
    long = geo_data[0].exterior.xy[0]
    lat = geo_data[0].exterior.xy[1]
    Map = folium.Map(location=[
        np.mean(lat), np.mean(long)], zoom_start=14, control_scale=True)

    for j in range(len(geo_data)):
        long = geo_data[j].exterior.xy[0]
        lat = geo_data[j].exterior.xy[1]
        folium.Marker([np.mean(lat), np.mean(long)], popup="<i>Mt. Hood Meadows</i>",
                      tooltip=f"type {cat[j] + 1}").add_to(Map)
    Map.save("map.html")