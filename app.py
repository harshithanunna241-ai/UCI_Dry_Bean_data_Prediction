import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("dry_bean_adaboost_model.pkl")

# Load label encoder
label_encoder = joblib.load("dry_bean_label_encoder.pkl")

# Page title
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱"
)

st.title("🌱 Dry Bean Classification")
st.write("AdaBoost Machine Learning Model")

st.subheader("Enter the Bean Details")

# Input features
Area = st.number_input("Area", min_value=0.0, value=50000.0)

Perimeter = st.number_input("Perimeter", min_value=0.0, value=800.0)

MajorAxisLength = st.number_input(
    "Major Axis Length",
    min_value=0.0,
    value=300.0
)

MinorAxisLength = st.number_input(
    "Minor Axis Length",
    min_value=0.0,
    value=200.0
)

AspectRation = st.number_input(
    "Aspect Ratio",
    min_value=0.0,
    value=1.5
)

Eccentricity = st.number_input(
    "Eccentricity",
    min_value=0.0,
    value=0.7
)

ConvexArea = st.number_input(
    "Convex Area",
    min_value=0.0,
    value=51000.0
)

EquivDiameter = st.number_input(
    "Equivalent Diameter",
    min_value=0.0,
    value=250.0
)

Extent = st.number_input(
    "Extent",
    min_value=0.0,
    value=0.75
)

Solidity = st.number_input(
    "Solidity",
    min_value=0.0,
    value=0.98
)

roundness = st.number_input(
    "Roundness",
    min_value=0.0,
    value=0.8
)

Compactness = st.number_input(
    "Compactness",
    min_value=0.0,
    value=0.7
)

ShapeFactor1 = st.number_input(
    "Shape Factor 1",
    min_value=0.0,
    value=0.006
)

ShapeFactor2 = st.number_input(
    "Shape Factor 2",
    min_value=0.0,
    value=0.002
)

ShapeFactor3 = st.number_input(
    "Shape Factor 3",
    min_value=0.0,
    value=0.5
)

ShapeFactor4 = st.number_input(
    "Shape Factor 4",
    min_value=0.0,
    value=0.99
)

# Prediction button
if st.button("Predict Bean Class"):

    input_data = pd.DataFrame([[
        Area,
        Perimeter,
        MajorAxisLength,
        MinorAxisLength,
        AspectRation,
        Eccentricity,
        ConvexArea,
        EquivDiameter,
        Extent,
        Solidity,
        roundness,
        Compactness,
        ShapeFactor1,
        ShapeFactor2,
        ShapeFactor3,
        ShapeFactor4
    ]], columns=[
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "AspectRation",
        "Eccentricity",
        "ConvexArea",
        "EquivDiameter",
        "Extent",
        "Solidity",
        "roundness",
        "Compactness",
        "ShapeFactor1",
        "ShapeFactor2",
        "ShapeFactor3",
        "ShapeFactor4"
    ])

    # Make prediction
    prediction = model.predict(input_data)

    # Convert numeric prediction to original class name
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Bean Class: {predicted_class}")