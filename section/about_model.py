import streamlit as st

st.title("About The Model: Att-ResRoBERTa")
st.divider()

# Introduction Section
st.subheader("🌟 Introduction")
st.write("""
    **Att-ResRoBERTa** is an advanced multimodal model designed for detecting **sarcasm** in code-switched **Tagalog-English** (Taglish) content. 
    By leveraging the strengths of both **text** and **image modalities**, the model aims to understand and interpret the nuanced expressions of sarcasm in 
    combined text-image pairs.
""")

# Problem Statement Section
st.subheader("❓ Problem Statement")
st.write("""
    Detecting sarcasm is a challenging task, especially in **code-switched languages** where the interplay between languages adds layers of complexity. 
    This task becomes even more intricate when both **textual** and **visual cues** are involved, as the model must discern incongruities between the two modalities.
""")

# Model Architecture Section
st.subheader("🛠️ Model Architecture")
st.write("""
    The **Att-ResRoBERTa** model integrates the following key components:
    
    - **Text Encoding**: 
        Utilizes the **XLM-RoBERTa** model to effectively encode the code-switched Taglish text data, capturing its linguistic features and nuances.
    
    - **Image Processing**: 
        Employs the **ResNet-50** architecture for analyzing accompanying images, extracting relevant visual features.
    
    - **Attention Mechanism**: 
        Incorporates an **attention-based approach** to highlight and align the incongruities between the text and image inputs, crucial for effective sarcasm detection.
""")

# XLM-RoBERTa Section
st.subheader("📖 XLM-RoBERTa")
st.write("""
    **XLM-RoBERTa** is a state-of-the-art multilingual transformer model built on the **RoBERTa** architecture. Key features include:
    
    - **Multilingual Capability**: Trained on a diverse dataset across multiple languages, making it suitable for code-switched content.
    
    - **Contextual Understanding**: Provides deep contextualized embeddings, enhancing the model's understanding of linguistic nuances in Taglish text.
    
    - **Transfer Learning**: Utilizes the pre-trained knowledge to improve performance on downstream tasks, such as sarcasm detection.
""")

# ResNet-50 Section
st.subheader("🖼️ ResNet-50")
st.write("""
    **ResNet-50** is a deep convolutional neural network renowned for its effectiveness in image recognition tasks. Notable aspects include:
    
    - **Residual Learning**: Introduces skip connections to help with the vanishing gradient problem, enabling the training of very deep networks.
    
    - **Feature Extraction**: Efficiently captures hierarchical features from images, making it adept at identifying visual cues related to sarcasm.
    
    - **Wide Applicability**: Successfully applied in various domains, including object detection and image classification, enhancing the model's robustness in multimodal scenarios.
""")

# Training Data Section
st.subheader("📚 Training Data")
st.write("""
    The model was trained on a carefully curated dataset consisting of **text-image pairs** that exhibit sarcastic and non-sarcastic annotations. 
    The dataset includes diverse examples of Taglish content, ensuring that the model generalizes well across different contexts and expressions.
""")

# Evaluation Metrics Section
st.subheader("📊 Evaluation Metrics")
st.write("""
    The performance of **Att-ResRoBERTa** is evaluated using several key metrics, including:
    
    - **Accuracy**: Measures the overall correctness of the model's predictions.
    
    - **Precision**: Indicates the proportion of true positive predictions among all positive predictions.
    
    - **Recall**: Assesses the model's ability to identify all relevant instances of sarcasm.
    
    - **F1 Score**: Provides a balance between precision and recall, giving a comprehensive measure of model performance.
""")

