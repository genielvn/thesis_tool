import streamlit as st
import pandas as pd
import json
from model.test_model import test_model, start_model
from PIL import Image

# Store model inside running streamlit
device, model, encoder, tokenizer = start_model()

st.image("./images/runmodelcover.jpg")
st.title("🤖Att-ResRoBERTa")


col1, col2 = st.columns(2)

# Image Modality Input Section
with col1:
    st.header("Image Modality")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)
    else:
        st.info("Please upload an image to continue.")

# Text Modality Input Section
with col2:
    st.header("Text Modality")
    txt = st.text_area(
        "Enter Code-Switched Text:",
        placeholder="Type your Taglish text here..."
    )
    st.write(f"📝 **Character count**: {len(txt)} characters.")

sarcastic = st.checkbox("Sarcastic")

# Submit Button to Run the Model
if st.button("Run Model", use_container_width=True):
    if uploaded_file is None or txt.strip() == "":
        st.warning("Both an image and text input are required to run the model.")
    else:
        test_model(device, model, encoder, tokenizer, txt, image, sarcastic)

# Divider for visual separation
st.divider()

with open("./model/json/bert.json", 'r') as file:
    bert_data = json.load(file)

with open("./model/json/roberta.json", 'r') as file:
    roberta_data = json.load(file)

# Sample data for comparison
data = {
    'Metric': ['Precision', 'Recall', 'F-Measure', 'Accuracy'],
    'Att-ResRoBERTa': [roberta_data["precision"], roberta_data["recall"], roberta_data["f_score"], roberta_data["test_accuracy"]],  # Replace with actual values
    'Att-BERT': [bert_data["precision"], bert_data["recall"], bert_data["f_score"], bert_data["test_accuracy"]],  # Replace with actual values
}

# Create a DataFrame
comparison_df = pd.DataFrame(data)

# Streamlit headers
st.header("Model Comparison")

# Display comparison table
st.dataframe(comparison_df, hide_index = True)  # Display the full table with both models