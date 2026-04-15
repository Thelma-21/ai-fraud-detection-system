import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# App configuration
st.set_page_config(page_title="💳 Fraud Detection", layout="wide")
st.title("💳 Fraud Detection System")

# Load and balance dataset
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")

    # Separate majority (legitimate) and minority (fraud) classes
    df_majority = df[df.Class == 0]
    df_minority = df[df.Class == 1]

    # Upsample minority class to balance dataset
    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    # Combine both classes
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    return df, df_balanced

# Train machine learning model
@st.cache_resource
def train_model(df_balanced):
    X = df_balanced.drop("Class", axis=1)
    y = df_balanced["Class"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest classifier
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, y_pred, X.columns.tolist()

# Load data and train model
with st.spinner("Loading data and training model (first run only)..."):
    df, df_balanced = load_data()
    model, X_train, X_test, y_train, y_test, y_pred, feature_cols = train_model(df_balanced)

st.success("Model ready!")

# Sidebar input section
st.sidebar.header("Check a Transaction")
st.sidebar.write("Enter the transaction details below.")

amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Or test with a real sample")

sample_type = st.sidebar.radio(
    "Sample transaction type",
    ["Random legitimate", "Random fraudulent"]
)

# Load sample transaction
if st.sidebar.button("Load sample transaction"):
    if sample_type == "Random legitimate":
        sample = df[df.Class == 0].sample(1)
    else:
        sample = df[df.Class == 1].sample(1)

    st.session_state["sample_row"] = sample

# Prediction logic
if st.sidebar.button("Check Transaction"):
    if "sample_row" in st.session_state:
        row = st.session_state["sample_row"]
        input_data = row[feature_cols].values
        true_label = row["Class"].values[0]
    else:
        input_data = [[amount] + [0] * (len(feature_cols) - 1)]
        true_label = None

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    st.sidebar.markdown("---")

    if prediction[0] == 1:
        st.sidebar.error(f"🚨 Fraud Detected!\nProbability: {prob[0][1]:.2%}")
    else:
        st.sidebar.success(f"✅ Legitimate Transaction\nProbability: {prob[0][0]:.2%}")

    # Display actual label when using sample data
    if true_label is not None:
        label_str = "Fraud" if true_label == 1 else "Legitimate"
        st.sidebar.info(f"Actual label: **{label_str}**")

# Model performance section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df)

with col2:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# Dataset class distribution
st.subheader("Dataset Class Distribution")
dist = df["Class"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
st.bar_chart(dist)
