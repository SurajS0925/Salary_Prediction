import streamlit as st
import pickle
import numpy as np
import json
from streamlit_lottie import st_lottie

def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


money = load_lottiefile("Animation - 1711601844985.json")
work = load_lottiefile("Animation - 1711601699581.json")
data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction :)")
    st_lottie(
        work,
        speed=1,
        reverse=False,
        loop=True,
        height=300,
        width=300
    )
    st.write("""### We need some information to predict the salary""")
    countries = (
        "United States of America",
        "Germany",
        "United Kingdom of Great Britain and Northern Ireland",
        "Canada",
        "India",
        "France",
        "Netherlands",
        "Australia",
        "Brazil",
        "Spain",
        "Sweden",
        "Italy",
        "Poland",
        "Switzerland",
        "Denmark",
        "Norway",
        "Israel",
    )
    country2=["United States of America",
              "Germany",
              "United Kingdom of Great Britain and Northern Ireland",
              "Canada",
            "India",
        "France",
        "Netherlands",
        "Australia",
        "Brazil",
        "Spain",
        "Sweden",
        "Italy",
        "Poland",
        "Switzerland",
        "Denmark",
        "Norway",
        "Israel"]

    symbol=["Dollars","the Euro","the pound sterling","Canadian dollars","Rupees","Euro","Euro","Australian dollars","Brazilian real","euro","Swedish krona","Euro","Polish złoty", "Swiss Franc","Danish Krone","Norwegian Krone"," Israeli New Shekel"]
    val=[1.0,0.92,0.79,1.36,83.34,0.92,0.92,1.53,4.99,0.92,10.62,0.92,3.99,0.9,6.89,10.79,3.68]
    # print(len(symbol),len(val))
    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    s="Dollars"
    cou=1.0
    for i in range(0,len(country2)):
        if country2[i]==country:
            s=symbol[i]
            cou=val[i]
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 1)

    ok = st.button("Predict Salary")
    if ok:
        X = np.array([[country, education, expericence]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st_lottie(
            money,
            speed=1,
            reverse=False,
            loop=True,
            height=300,
            width=300
        )
        st.subheader(f"The Predicted/Estimated salary in Dollars is ${salary[0]:.2f} per annum")
        sal=cou*salary
        st.subheader(f"The Predicted/Estimated salary in {s} is {sal[0]:.2f} per annum")