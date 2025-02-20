import streamlit as st
import random

st.set_page_config(page_title="Text Source Identifier", layout="centered")
st.title("Text Source Identifier")
classifier_tab, report_tab = st.tabs(["Classifier", "Report"])

with classifier_tab:
    st.header("Distinguish Human vs. Machine Written Text")
    input_text = st.text_area("Enter your text here:")

    if st.button("Classify"):
        if input_text.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            # Randomly choose a prediction for demonstration purposes
            result = random.choice(["Human Written Text", "Machine Generated Text"])
            st.success(f"Prediction: {result}")

with report_tab:
    st.header("Report")
    with open("assets/report.md", "r") as file:
        report = file.read()
    st.markdown(report)
