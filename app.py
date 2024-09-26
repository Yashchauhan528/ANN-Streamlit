import streamlit as st
import numpy as np
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder 
import pickle as pkl

model = tf.keras.models.load_model('./my_model.keras')

with open('LabelEncoder_gender.pkl','rb') as file:
    le_gender = pkl.load(file)
    
with open('onh_geo.pkl','rb') as file:
    onh_geo = pkl.load(file)
    
with open('scaler.pkl','rb') as file:
    sc = pkl.load(file)
    
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography',onh_geo.categories_[0])
gender = st.selectbox('Gender',le_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',0,10)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = {
    'CreditScore':credit_score,
    'Gender':le_gender.transform([gender])[0],
    'Age':age,
    'Tenure':tenure,
    'Balance':balance,
    'NumOfProducts':num_of_products,
    'HasCrCard':has_cr_card,
    'IsActiveMember':is_active_member,
    'EstimatedSalary':estimated_salary
}

geo_encoded = onh_geo.transform([[geography]]).toarray()

geo_df = pd.DataFrame(geo_encoded,columns=onh_geo.get_feature_names_out())

input_df = pd.DataFrame([input_data])

df = pd.concat([input_df,geo_df],axis=1)

final_data = sc.transform(df)

prediction = model.predict(final_data)

prediction_proba = prediction[0][0]

st.write(f"Customer Churn Probability : {np.round(prediction_proba*100,2)}%")

if prediction_proba < 0.5:
    st.write('Customer is not likely to churn')
else:
    st.write('Customer is likely to churn')