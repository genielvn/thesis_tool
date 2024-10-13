import streamlit as st
st.image("./images/aboutmodelcover.jpg")
st.title("🤖 About The Model: Att-ResRoBERTa")
st.divider()



# Problem Statement Section
st.subheader("🌟 Introduction")
st.write("""
    Detecting sarcasm is a challenging task, especially in **code-switched languages** where the interplay between languages adds layers of complexity. 
    This task becomes even more intricate when both **textual** and **visual cues** are involved, as the model must discern **incongruities** between the two modalities.

    Despite remarkable progress made in **multi-modal sarcasm detection**, to the best of our knowledge, existing studies (Cai, Cai, & Wan, 2019) (Fang, Liang, & Xiang, 2024) (Liang B. , et al., 2022) (Liu, Wang, & Li, 2022) (Pan, Lin, Fu, Qi, & Wang, 2020) (Xu, Zeng, & Mao, 2020) tend to implement it in :blue-background[**high resource languages like English**]. However, in social media posts, there are a huge number of **Filipino users** that use **code-switching text**, particularly **Tagalog-English** text as their medium. **Code-switching** is a term that alternates the use of two or more languages in a single conversation or text (Morisson, 2024). 

    :blue-background[**Low-resource languages**] like **Filipino** are :blue-background[**suffering from the lack of datasets and pre-trained models**]. The models we usually use can expose hidden weaknesses and the model’s true capability when datasets from low-resource languages are used (Cruz & Cheng, 2020). :blue-background[**To give models some improvements, more datasets must be created to train it**]. 

    - Hence, we will be proposing a novel **Att-ResRoBERTa** to cater to the **multimodal Sarcasm Detection** in code-switched Tagalog-English with text-image pairs dataset in Tagalog-English. 
    - We will be utilizing the model architecture of (Pan, Lin, Fu, Qi, & Wang, 2020) that draws support from the **self-attention mechanism** to model the **incongruity** between text and image, using **query**, **key**, **value**, and modifying it by removing the **intra-modal incongruity** between text and **hashtag** since hashtags can cause bias (Qin, et al., 2023). 
    - Afterwards, fine-tuning the **XLM-RoBERTa** will be done for the **multimodal sarcasm detection task**, a variant of the **BERT model**, that is capable of handling code-switching within multiple languages.
""")

# Introduction Section
st.subheader("❓ Problem Statement")
st.write("""
1. What is the proposed **Att-ResRoBERTa** model's performance in detecting **code-switched Tagalog-English Multimodal Sarcasm** in text-image?
    
    a. **Precision**  
    b. **Recall**  
    c. **F-Measure**  
    d. **Accuracy**  

2. What is the significant difference between the **BERT-based architecture model** by (Pan, Lin, Fu, Qi, & Wang, 2020) and the proposed **Att-ResRoBERTa** model's performance in detecting **Multimodal Sarcasm** in **Code-Switched Tagalog-English** contents?
""")



# Model Architecture Section
st.subheader("🛠️ Model Architecture")
st.image("./images/system_arch.png")
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

