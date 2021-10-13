import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import geopandas
import plotly.express as px 
from datetime import datetime, time
from PIL import Image
image=Image.open('house_hocket.png')
st.sidebar.image(image,use_column_width=True)
############################################################
#FUNCOES
############################################################

@st.cache(allow_output_mutation=True)
def get_data(file):
    data = pd.read_csv(file).sample(1000)
    return data

@st.cache(allow_output_mutation=True)
def geo_file(url):
    geofile = geopandas.read_file(url)
    return geofile

def stats_descriptive(file):
    st.title('Data Overview')
    data = get_data(file)
    st.dataframe(data)
    ## Average Metrics

    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['zipcode', 'total house', 'price', 'sqrt living', 'price_m2']

    c1, c2 = st.columns((1,1))

    c1.title('Average Metrics')
    c1.dataframe(df)

    num_features = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_features.apply(np.mean))
    mediana = pd.DataFrame(num_features.apply(np.median))
    std = pd.DataFrame(num_features.apply(np.std))

    maxi = pd.DataFrame(num_features.apply(np.max))
    mini = pd.DataFrame(num_features.apply(np.min))

    df_statis = pd.concat([maxi, mini, media, mediana, std], axis=1).reset_index()
    df_statis.columns = ['features', 'max', 'min', 'media', 'mediana', 'std']
    c2.title('Análise Descritiva')
    c2.dataframe(df_statis)
    return None

def set_feature(data):
    data['sqm_lot']= data['sqft_lot']/10.764
    data['sqm_living']=data['sqft_living']/10.764
    data['price_m2']=data['price']/data['sqm_lot']
    data['date']=pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    return data

def mapas(data, geofile):
    st.header('Portfólio Density')

    # Base Map = Folium
    density_map=folium.Map(location=[data['lat'].mean(),
                           data['long'].mean()],
                           default_zoom_start=15)
    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in data.iterrows():
        folium.Marker([row['lat'],row['long']],
        popup='Sold R${0} on: {1} Sqm Living: {2} Bedrooms: {3} Bathrooms: {4}   Year Built: {5}'.format(
            row['price'],
            row['date'],
            row['sqm_living'],
            row['bedrooms'],
            row['bathrooms'],
            row['yr_built'])).add_to(marker_cluster)
        
    folium_static(density_map)
    
    # Region Price Map
    st.header('Price Density')
    
    df= data[['price','zipcode']].groupby('zipcode').mean().reset_index()
    df.columns= ['ZIP','PRICE']
        
    geofile=geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]
    
    region_price_map= folium.Map(location=[data['lat'].mean(),
                            data['long'].mean()],
                            default_zoom_start=15)
    
    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP','PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity = 0.7,
                                line_opacity = 0.2,
                                legend_name='AVG PRICE')
    
        
    folium_static(region_price_map)
    return None    

def visualizacao(data):
    st.sidebar.title( 'Commercial Options' )
    st.title( 'Commercial Attributes' )

       # ---------- Average Price per year built
    # setup filters
    min_year_built = int( data['yr_built'].min() )
    max_year_built = int( data['yr_built'].max() )

    st.sidebar.subheader( 'Select Max Year Built' )
    f_year_built = st.sidebar.slider( 'Year Built', min_year_built, max_year_built, min_year_built )

    st.header( 'Average price per year built' )

    # get data
    data['date'] = pd.to_datetime( data['date'] ).dt.strftime( '%Y-%m-%d' )

    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby( 'yr_built' ).mean().reset_index()

    fig = px.line( df, x='yr_built', y='price' )
    st.plotly_chart( fig, use_container_width=True )


    # ---------- Average Price per day
    st.header( 'Average Price per day' )
    st.sidebar.subheader( 'Select Max Date' )

    # setup filters
    min_date = datetime.strptime( data['date'].min(), '%Y-%m-%d' )
    max_date = datetime.strptime( data['date'].max(), '%Y-%m-%d' )

    f_date = st.sidebar.slider( 'Date', min_date, max_date, min_date )

    # filter data
    data['date'] = pd.to_datetime( data['date'] )
    df = data[data['date'] < f_date]
    df = df[['date', 'price']].groupby( 'date' ).mean().reset_index()

    fig = px.line( df, x='date', y='price' )
    st.plotly_chart( fig, use_container_width=True )
    # ---------- Histogram -----------
    st.header( 'Price Distribuition' )
    st.sidebar.subheader( 'Select Max Price' )

    # filters
    min_price = int( data['price'].min() )
    max_price = int( data['price'].max() )
    avg_price = int( data['price'].mean() )

    f_price = st.sidebar.slider( 'Price', min_price, max_price, avg_price )

    df = data[data['price'] < f_price]

    fig = px.histogram( df, x='price', nbins=50 )
    st.plotly_chart( fig, use_container_width=True )
    return None

def set_phisical( data ):
    st.sidebar.title( 'Attributes Options' )
    st.title( 'House Attributes' )

    # filters
    f_bedrooms = st.sidebar.selectbox( 'Max number of bedrooms', 
                                        sorted( set( data['bedrooms'].unique() ) ) )
    f_bathrooms = st.sidebar.selectbox( 'Max number of bath', 
                                        sorted( set( data['bathrooms'].unique() ) ) )

    c1, c2 = st.columns( 2 )

    # Houses per bedrooms
    c1.header( 'Houses per bedrooms' )
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram( df, x='bedrooms', nbins=19 )
    c1.plotly_chart( fig, use_containder_width=True )

    # Houses per bathrooms
    c2.header( 'Houses per bathrooms' )
    df = data[data['bathrooms'] < f_bathrooms]
    fig = px.histogram( df, x='bathrooms', nbins=10 )
    c2.plotly_chart( fig, use_containder_width=True )

    # filters
    f_floors = st.sidebar.selectbox('Max number of floors', sorted( set( data['floors'].unique() ) ) )
    f_waterview = st.sidebar.checkbox('Only House with Water View' )

    c1, c2 = st.columns( 2 )

    # Houses per floors
    c1.header( 'Houses per floors' )
    df = data[data['floors'] < f_floors]
    fig = px.histogram( df, x='floors', nbins=19 )
    c1.plotly_chart( fig, use_containder_width=True )

    # Houses per water view
    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    fig = px.histogram( df, x='waterfront', nbins=10 )
    c2.header( 'Houses per water view' )
    c2.plotly_chart( fig, use_containder_width=True )

    return None

if __name__ == '__main__':
    ##get data
    file = '/Users/tuanymariah/portfolio/kc_house_data.csv'
    geofile = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    data = get_data(file)
    geofile = geo_file(geofile)
    dataset = set_feature(data)


    op_home = ('Home',
               'Estatística Descritiva',
               'Densidade de Portifólio',
               'Visualização de Dados',
               'Insights de Mercado',
               'Avaliação Imobiliária')
    opcoes = st.sidebar.radio( "Qual seção do Projeto deseja visitar? ",op_home)
    if opcoes == 'Home':
        st.write('Inserir as Coisa')
    elif opcoes == 'Estatística Descritiva':
        data = stats_descriptive(file)
    elif opcoes == 'Densidade de Portifólio':
        mapas(dataset, geofile)
    elif opcoes == 'Visualização de Dados':
        visualizacao(dataset)
        set_phisical(dataset)
    elif opcoes == 'Insights de Mercado':
        st.write('colocar coisas sobre insisght de mercado ')
    elif opcoes == 'Avaliação Imobiliária':
        st.write('colocar coisas sobre avaliacao imobiliaria de mercado ')



