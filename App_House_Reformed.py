#House Rocket App

import pandas as pd
import streamlit as st
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import geopandas
import plotly.express as px
from datetime import datetime

#Settings
pd.set_option('display.float_format', lambda x: '%.2f' % x)
st.set_page_config(layout='wide')

#Helper functions
@st.cache(allow_output_mutation=True)
def load_data(path):
    data = pd.read_csv(path)
    return data

def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile

def set_attributes(data):
    data['price_m2'] = data['price']/(data['sqft_lot']*0.092903)
    data.drop(columns=['sqft_living15', 'sqft_lot15'])
    data['yr_renovated'] = pd.to_datetime(data['yr_renovated']).dt.strftime('%Y')
    data['waterfront'] = data.waterfront.apply(lambda x: 'no' if x == 0 else 'yes')
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    #data['yr_built'] = pd.to_datetime(data['yr_built']).dt.strftime('%Y')
    return data

def data_overview(data):
    f_atribbutes = st.sidebar.multiselect('Enter columns', data.columns)
    f_zipcode = st.sidebar.multiselect("Enter with zipcode", data.zipcode.unique())

    st.title('Data Overview of House Rocket')
    if (f_atribbutes != []) & (f_zipcode !=[]):
        choosed = data.loc[data.zipcode.isin(f_zipcode), f_atribbutes]
    elif (f_atribbutes == []) & (f_zipcode !=[]):
        choosed =data.loc[data['zipcode'].isin(f_zipcode), :]
    elif (f_atribbutes != []) & (f_zipcode == []):
        choosed =data.loc[:, f_atribbutes]
    else:
        choosed = data.copy()

    #Average metrics:
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    #merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    m3 = pd.merge(m2, df4, on='zipcode', how='inner')
    m3.columns = ['Zipcode', 'ID', 'Mean Price', 'Mean Living room', 'Mean price/m2']

    #Statistic Descriptive
    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    dt1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
    dt1.columns = ['Atributtes', 'Max', 'Min', 'Mean', 'Median', 'STD']

    #Load Filters
    #f_atribbutes
    #f_zipcode

    #load data Views
    st.header('Overview data')
    st.dataframe(choosed.head())
    c1, c2 = st.beta_columns((1,1))
    c1.header('Data Atributtes')
    c1.dataframe(m3, height=600)
    c2.header('Data statistc analyses')
    c2.dataframe(dt1, height = 600)
    return None

def regio_overview(data,geofile):
    st.title('Region Overview')
    c1, c2 = st.beta_columns((1,1))
    c1.header('Portifolio Density')

    df = data.sample(10)

    #Base Map - Folium
    density_map = folium.Map(location=[data['lat'].mean(),
                            data['long'].mean()],
                        default_zoom_start=15)


    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                    popup='Sold R${0} on: {1}, Features: {2}, sqft, {3} bedrooms,'
                            '{4} bathrooms, year built: {5}'. format(row['price'],
                                                                    row['date'],
                                                                    row['sqft_living'],
                                                                    row['bedrooms'],
                                                                    row['bathrooms'],
                                                                    row['yr_built'])).add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    #Region Price Map
    c2.header('Price Density')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns=['ZIP', 'PRICE']

    df = df.sample(10)

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location =[data['lat'].mean(),
                                            data['long'].mean()],
                                            default_zoom_start=15)

    region_price_map.choropleth(data=df,
                            geo_data=geofile,
                            columns=['ZIP', 'PRICE'],
                            key_on='feature.properties.ZIP',
                            fill_color ='YlOrRd',
                            fill_opacity=0.7,
                            line_opacity=0.2,
                            legend_name='AVG PRICE')

    with c2:
        folium_static(region_price_map)

    return None

def set_commercial(data):
    # ===== Distribuição por categorias comerciais ====
    st.sidebar.title("Comercial Option")
    st.title('Comercial Attributes')

    #------Average price per Year---------
    #filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built', min_year_built,
                                    max_year_built,
                                    min_year_built)

    st.header('Average Price per Year built')
    #data selection
    df = data.loc[data['yr_built']< f_year_built]
    df = df[['yr_built', 'price']].groupby(data['yr_built']).mean()

    #Plot
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    #------Average price per Day---------
    st.header('Average Price per Day')
    st.sidebar.subheader('Select Max Date')

    #filters
    #data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')
    #st.write(type(min_date))

    f_date = st.sidebar.slider('Date', min_date, max_date, min_date)

    #data filtering
    #st.write(type(f_date))
    data['date'] = pd.to_datetime(data['date'])
    #st.write(type(data['date']))
    df = data.loc[data['date']<f_date]
    df = df[['date', 'price']].groupby(data['date']).mean().reset_index()

    #Plot
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    #------Histogram ---------
    st.header('Price Distibution')
    st.sidebar.subheader('Select Max Price')

    #filter
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    #data filtering
    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)
    df = data.loc[data['price']<f_price]

    #data plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    #------ Distribuição dos imóveis por categorias físicas ---------
    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    #filters
    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms',
                                    sorted(set(data['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox('Max number of bathrooms',
                                    sorted(set(data['bathrooms'].unique())))

    c1, c2 = st.beta_columns(2)
    #House per bedrooms
    c1.header('Houses per bedrooms')
    df = data[data['bedrooms']<f_bedrooms]
    fig = px.histogram(data, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    #House per bathrooms
    c2.header('Houses per bathrooms')
    df = data[data['bedrooms']<f_bathrooms]
    fig = px.histogram(data, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    #filters
    f_floors = st.sidebar.selectbox('Max number of floor',
                                    sorted(set(data['floors'].unique())))
    f_waterfront = st.sidebar.checkbox('Only Houses with Water view')

    c1, c2 = st.beta_columns(2)

    #House per floors
    c1.header('Houses per floors')
    df = data[data['floors']<f_floors]
    #plot
    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    #House per water view
    c2.header('Houses with waterfront')
    if f_waterfront:
        df=data[data['waterfront']=='yes']
    else:
        df=data.copy()

    #plot
    fig = px.histogram(df, x='waterfront', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)
    return None

if __name__ == "__main__":
    #get data
    path = 'kc_house_data.csv'
    url = 'http://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    #load data
    data = load_data(path)
    geofile = get_geofile(url)

    #transform data features
    data = set_attributes(data)
    data_overview(data)
    regio_overview(data, geofile)
    set_commercial(data)
    
    #Load data maps
    #We dont have a data set

