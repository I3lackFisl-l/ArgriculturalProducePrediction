
import streamlit as st
import plotly.graph_objects as px
import numpy as np
import pandas as pd
import pickle
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder,RobustScaler, LabelEncoder
from dataPrep.datapreparation import DataPreparation
# from dataPrep import DataPreparation


# Load models
model_path = 'models/V1.1'
all_around_model = pickle.load(
    open(model_path + '/argriculture-produce-prediction-v.1.1.sav', 'rb'))
durian_model = pickle.load(
    open(model_path + '/durian-prediction-v.1.1.sav', 'rb'))
longan_model = pickle.load(
    open(model_path + '/longan-prediction-v.1.1.sav', 'rb'))
lychees_model = pickle.load(
    open(model_path + '/lychees-prediction-v.1.1.sav', 'rb'))
mangosteen_model = pickle.load(
    open(model_path + '/mangosteen-prediction-v.1.1.sav', 'rb'))
rambutan_model = pickle.load(
    open(model_path + '/rambutan-prediction-v.1.1.sav', 'rb'))
wollongong_model = pickle.load(
    open(model_path + '/wollongong-prediction-v.1.1.sav', 'rb'))

# Load Data Preparation
data_prep_path = 'dataPrep'
# dataPrep = pickle.load(open(data_prep_path + '/data.prepare.dataPrep.sav','rb'))
dataPrep = DataPreparation()
# encoder = OneHotEncoder()

normalize_and_transforms = pickle.load(open(data_prep_path + '/data.prepare.normalizeAndTransform.sav','rb'))

st.set_page_config(
    page_title="Argricultural Prediction",
    layout="wide"
)

st.title("ระบบทำนายผลผลิตทางการเกษตรจากปริมาณน้ำฝน")

current_date = datetime.date.today()
month = {1: 'มกราคม', 2: 'กุมภาพันธ์', 3: 'มีนาคม', 4: 'เมษายน', 5: 'พฤษภาคม', 6: 'มิถุนายน',
         7: 'กรกฎาคม', 8: 'สิงหาคม', 9: 'กันยายน', 10: 'ตุลาคม', 11: 'พฤจิกายน', 12: 'ธันวาคม'}
year = np.arange(start=current_date.year-5, stop=current_date.year+5, step=1)
df_province = pd.read_csv('sources/thai_provinces.csv')

def format_func(option):
    return month[option]

plant = st.radio(
    "ประเภทผลผลิต",
    ('เงาะ', 'ลำไย', 'มังคุด', 'ลองกอง', 'ทุเรียน', 'ลิ้นจี่'), horizontal=True)

prov_value = st.selectbox('จังหวัด', df_province['name_th'])
month_no = st.selectbox('เดือน', options=list(month.keys()), format_func=format_func, index=current_date.month-1)
year_no = st.selectbox('ปี', year, index=5)
min_max_rain = st.slider(
    'ปริมาณนำ้ฝนต่ำสุดและสูงสุด',
    0.0, 1500.0, (100.0, 400.0))
avg_rain = st.slider(
    'ปริมาณนำ้ฝนโดยเฉลี่ย',
    0.0, 1500.0, 300.0)

#set data
df = pd.DataFrame(data=np.array([plant, prov_value, min_max_rain[0], min_max_rain[1], avg_rain, year_no, month_no]).reshape(1,-1), 
                columns=['plant', 'province_name', 'min_rain', 'max_rain','avg_rain', 'year_no', 'month_no'])
df = dataPrep.castType(df)

def selectFruitModel(fruit):
    if fruit == "เงาะ":
        return rambutan_model
    elif fruit == "ลำไย":
        return longan_model
    elif fruit == "มังคุด":
        return mangosteen_model
    elif fruit == "ลองกอง":
        return wollongong_model
    elif fruit == "ทุเรียน":
        return durian_model
    elif fruit == "ลิ้นจี่":
        return lychees_model


#make a prediction
aam_pred_val = all_around_model.predict(df)
df_fruit = df.drop(['plant'], axis=1)
f_pred_val = selectFruitModel(plant).predict(df_fruit)
# f_pred_val = durian_model.predict(df_fruit)
#display prediction value
st.markdown("#### ผลผลิตการทำนายจากแบบจำลองรวมทุกประเภทผลผลิต  : "+  str(f'{aam_pred_val[0]:.2f}') )
st.markdown("#### ผลผลิตการทำนายจากแบบจำลองเฉพาะประเภท" + plant + " : "+  str(f'{f_pred_val[0]:.2f}'))

#https://i3lackfisl-l-argriculturalproduceprediction-app-s11bs3.streamlit.app/