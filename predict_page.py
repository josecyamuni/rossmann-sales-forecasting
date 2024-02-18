import streamlit as st
import pickle
import numpy as np
from datetime import date
import pandas as pd

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data["model"]
scaler = data["scaler"]
store_df = data["store_df"]

def show_predict_page():

    def check_promo_month(promoInterval, month, promo2open):
        month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                    7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

        try:
            months = (promoInterval or '').split(',')
            if promo2open and month2str[month] in months:
                return 1
            else:
                return 0
        except Exception:
            return 0
    
    st.title("Rossmann Stores Sales Daily Prediction")

    st.write("""### We need some information to predict the sales in your store""")

    store = st.selectbox("Select the Store ID:", range(1, 1116))

    fecha_limite_inferior = date(2015, 1, 1)
    fecha_limite_superior = date(2016, 3, 1)

    selected_date = st.date_input("Selecciona una fecha", 
                                  min_value=fecha_limite_inferior, 
                                  max_value=fecha_limite_superior,
                                  value=fecha_limite_inferior)

    st.write("""Is there going to be a promotion that day?""")

    promo = None

    clicked_yes = st.button("Yes")
    clicked_no = st.button("No")

    if clicked_yes:
        promo = 1
    else:
        promo = 0


    competitionDistance = store_df.loc[store_df['Store'] == store, 'CompetitionDistance'].values[0]
    if pd.isna(competitionDistance):
        max_distance = 100000  
        competitionDistance = max_distance

    promo2 = store_df.loc[store_df['Store'] == store, 'Promo2'].values[0]
    
    assorment = store_df.loc[store_df['Store'] == store, 'Assortment'].values[0]

    if assorment == 'c':
        assort_c = 1
    else:
        assort_c = 0

    storeType = store_df.loc[store_df['Store'] == store, 'StoreType'].values[0]
    if storeType == 'b':
        st_b, st_c , st_d = 1, 0, 0
    elif storeType == 'c':
        st_b, st_c , st_d = 0, 1, 0
    elif storeType == 'd':
        st_b, st_c , st_d = 0, 0, 1
    else:
        st_b, st_c , st_d = 0, 0, 0        

    current_date = pd.to_datetime(selected_date)
    year = current_date.year
    month = current_date.month
    day = current_date.day
    dayOfWeek = current_date.weekday() + 1
    weekOfYear = current_date.isocalendar()[1]

    comp_since_year = store_df.loc[store_df['Store'] == store, 'CompetitionOpenSinceYear']
    comp_since_month = store_df.loc[store_df['Store'] == store, 'CompetitionOpenSinceMonth']

    compOpen = 12 * (year - comp_since_year) + (month - comp_since_month)
    compOpen = float(compOpen.map(lambda x: 0 if x < 0 else x).fillna(0))

    promo2_since_year = store_df.loc[store_df['Store'] == store, 'Promo2SinceYear']
    promo2_since_week = store_df.loc[store_df['Store'] == store, 'Promo2SinceWeek']

    promo2open = 12 * (year - promo2_since_year) + (weekOfYear - promo2_since_week)*7/30.5
    promo2open = float(promo2open.map(lambda x: 0 if x < 0 else x).fillna(0)*promo2)

    promoInterval = store_df.loc[store_df['Store'] == store, 'PromoInterval'].values[0]
    isPromo2month = check_promo_month(promoInterval, month, promo2open)


    ok = st.button("Predict sales")
    
    if ok:

        # numeric_data
        nd = np.array([[store, dayOfWeek, promo, competitionDistance, 
                        promo2, year, month, day, weekOfYear, 
                        compOpen, promo2open, isPromo2month]])
        # scaled numeric data
        snd = scaler.transform(nd)
        # categorical data
        cd = np.array([[st_b, st_c, st_d, assort_c]])

        combined_data = np.concatenate([snd, cd], axis=1)

        sales = model.predict(combined_data)
        st.subheader(f"The estimated amount of sales in the store {store} on the day {current_date.strftime('%Y-%m-%d')} is ${sales[0]:.2f}")