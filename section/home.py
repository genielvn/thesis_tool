import streamlit as st

# Title and Author Section
st.image("./images/aboutmodelcover.jpg")
st.title("Code-switched Tagalog-English Text-Image Multimodal Sarcasm Detection Using Attention Mechanism for Inter-modal Incongruity Modeling")
st.caption("**Authors**: Kim Montana, Genesis Lovino, Monica Zapanta, Bien Gavino, Lander Pardilla")
st.divider()

# Welcome Section
st.header("👋 Welcome to Our Tool!")
st.write("""
This tool is designed to demonstrate **multimodal sarcasm detection** for **code-switched Tagalog-English (Taglish)** text and image pairs.
By leveraging an **attention mechanism**, it effectively captures sarcasm through incongruities between text and images.
""")

# Introduction
st.subheader("🔍 Introduction")
st.write("""
The primary goal of this tool is to detect sarcasm in text-image pairs, focusing on the nuanced interplay between text and visuals. 
It is specially tailored to handle the **Taglish** code-switching common in social media and other informal contexts.
""")

# How it Works
st.subheader("⚙️ How It Works")
st.markdown("""
Our model architecture includes:
- **Text Encoding**: Uses the **XLM-RoBERTa** model to process Taglish code-mixed text data.
- **Image Processing**: Employs **ResNet-50** to analyze the accompanying image.
- **Multimodal Integration**: Utilizes an attention mechanism to detect mismatches or incongruities between the text and image, which are key signals for sarcasm detection.
""")

# Key Features
st.subheader("✨ Key Features")
st.markdown("""
- **Multimodal Fusion**: Combines both text and image inputs to deliver accurate sarcasm detection.
- **Code-switched Language Support**: Tailored to handle sarcasm in code-switched Taglish, combining both English and Tagalog seamlessly.
- **Attention-based Mechanism**: Focuses on subtle misalignments between the textual and visual elements to highlight sarcasm.
""")

# How to Use the Tool
st.subheader("🛠️ How to Use the Tool")
st.markdown("""
1. **Upload Text and Image Pair**: Provide a Taglish text along with an image.
2. **Run Sarcasm Detection**: The model will analyze the input and return a sarcasm probability score.
""")

# Footer Section for Contextual Information
st.divider()
st.caption("Polytechnic University of the Philippines")
