import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle


def add_prediction(input_values):
    # load the model from disk
    with open('model\logistic_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # load the scaler from disk
    with open('model\scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    input_df = pd.DataFrame([input_values])
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    if prediction[0] == 1:
        st.error("The model predicts that this is a Malignant tumor (Cancerous).")
    else:
        st.success("The model predicts that this is a Benign tumor (Non-Cancerous).")

    st.subheader("Prediction Probability")

    # Convert model probabilities into DataFrame
    proba_df = pd.DataFrame(prediction_proba, columns=['Benign', 'Malignant'])

    # Transpose so classes are rows, not columns
    proba_df = proba_df.T
    proba_df.columns = ["Probability"]

    # Display as a bar chart
    st.bar_chart(proba_df)

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def get_scaled_values(input_values):
    data = clean_data()

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_values.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_values):
    input_values = get_scaled_values(input_values)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_values['radius_mean'], input_values['texture_mean'], input_values['perimeter_mean'],
            input_values['area_mean'], input_values['smoothness_mean'], input_values['compactness_mean'],
            input_values['concavity_mean'], input_values['concave points_mean'], input_values['symmetry_mean'],
            input_values['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Values'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_values['radius_se'], input_values['texture_se'], input_values['perimeter_se'],
            input_values['area_se'], input_values['smoothness_se'], input_values['compactness_se'],
            input_values['concavity_se'], input_values['concave points_se'], input_values['symmetry_se'],
            input_values['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='standard error values'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_values['radius_worst'], input_values['texture_worst'], input_values['perimeter_worst'],
            input_values['area_worst'], input_values['smoothness_worst'], input_values['compactness_worst'],
            input_values['concavity_worst'], input_values['concave points_worst'], input_values['symmetry_worst'],
            input_values['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst values'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )

    return fig
def clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(columns=['Unnamed: 32', 'id'])
    # diagnosis column encode
    data["diagnosis"] = data["diagnosis"].map({'M': 1, 'B': 0})
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = clean_data()
    slider_labels = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 
        'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
        'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 
        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
        'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
        'concavity_worst', 'concave points_worst', 'symmetry_worst', 
        'fractal_dimension_worst'
    ]
    inputs_dict = {}
    for label in slider_labels:
        inputs_dict[label] = st.sidebar.slider(label, min_value=0.0, 
        max_value=float(data[label].max()), 
        value=float(data[label].mean()))

    return inputs_dict

def main():
    st.set_page_config(
    page_title="Breast Cancer Prediction", 
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
    )

    input_values = add_sidebar()
    
    with st.container():
        st.title("Breast Cancer Prediction Application")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")
        # move columns here
        col1, col2 = st.columns([4, 1])
        with col1:
           radar_chart = get_radar_chart(input_values)
           st.plotly_chart(radar_chart, use_container_width=True)
           st.image("https://cdn.prod.website-files.com/60995de2aeb0c37606ec3f7e/6537c56c122672ba0d18dbe1_Breast%20cancer%20awareness%20month_maine_OP1-p-1600.png"  , caption="Breast Cancer Awareness")
 
        with col2:
            add_prediction(input_values)








if __name__ == "__main__":
    main()

