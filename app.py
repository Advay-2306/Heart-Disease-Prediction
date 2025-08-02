import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Preprocessing ---
# Load the dataset (adjust path or use uploader for deployment)
df = pd.read_csv('heart_disease_uci.csv')  # Replace with URL or uploader if needed

# Preprocessing steps
df['target'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
numerical_cols = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())  # Fixed typo
categorical_cols = ['fbs', 'restecg', 'exang', 'slope', 'thal']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
df['fbs'] = df['fbs'].astype(int)
df['exang'] = df['exang'].astype(int)
df['chol'] = df['chol'].replace(0, df['chol'].median())
df['trestbps'] = df['trestbps'].replace(0, df['trestbps'].median())
df = pd.get_dummies(df, columns=['sex', 'cp', 'restecg', 'slope', 'thal'], drop_first=True)

# --- Model Training with SVM ---
features = [col for col in df.columns if col not in ['id', 'dataset', 'num', 'target']]
X = df[features]
y = df['target']
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(probability=True)
model.fit(X_train, y_train)

st.title("❤️ HEART DISEASE PREDICTION - ML Classification Project")
st.markdown("Enter your details below to get a prediction of your heart disease risk.")

st.sidebar.header("User Input Features")

def user_input_features():
    input_dict = {
        'age': st.sidebar.slider('Age', 29, 77, 54),
        'trestbps': st.sidebar.slider('Resting Blood Pressure', 94, 200, 132),
        'chol': st.sidebar.slider('Serum Cholestoral in mg/dl', 126, 564, 240),
        'thalch': st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 149),
        'oldpeak': st.sidebar.slider('ST depression induced by exercise', 0.0, 6.2, 1.0),
        'ca': st.sidebar.selectbox('Number of major vessels colored by fluoroscopy', (0, 1, 2, 3), 0),
        'sex': st.sidebar.selectbox('Sex', ('Male', 'Female')),
        'cp': st.sidebar.selectbox('Chest Pain Type',
                                  ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic')),
        'fbs': st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1), 0),
        'restecg': st.sidebar.selectbox('Resting Electrocardiographic Results',
                                       ('normal', 'st-t abnormality', 'lv hypertrophy')),
        'exang': st.sidebar.selectbox('Exercise Induced Angina', (0, 1), 0),
        'slope': st.sidebar.selectbox('Slope of the peak exercise ST segment', ('upsloping', 'flat', 'downsloping')),
        'thal': st.sidebar.selectbox('Thal', ('normal', 'fixed defect', 'reversible defect')),  # Fixed typo
    }
    df_input = pd.DataFrame(input_dict, index=[0])
    df_input = pd.get_dummies(df_input)
    for col in X.columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[X.columns]
    return df_input

input_df = user_input_features()
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Display user input
st.subheader("Your Input:")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction:")
if prediction[0] == 1:
    st.error("There is a high probability of heart disease.")
else:
    st.success("There is a low probability of heart disease.")

st.subheader("Prediction Probability:")
st.write(f"Probability of No Heart Disease: **{prediction_proba[0][0]:.2f}**")
st.write(f"Probability of Heart Disease: **{prediction_proba[0][1]:.2f}**")

# Add column mismatch check
missing_cols = [col for col in X.columns if col not in input_df.columns]
if missing_cols:
    st.error(f"Missing columns in input: {missing_cols}. Please check data alignment.")
    st.stop()