import streamlit as st
import matplotlib.pyplot as plt

# Title with an emoji
st.image("./images/datasetheader.jpg")
st.title("🔍 **Dataset Profile**: Code-Switched Tagalog-English Sarcasm Dataset")
st.divider()

# Introduction section with emojis
st.header("✨ Overview")
st.write("""
Welcome to the **dataset profile** of our model, which focuses on detecting sarcasm in image-text posts using Code-Switched Tagalog-English data. Here's a quick look at the data and steps involved in developing our **sarcasm detection system**. 🚀
""")

st.subheader("📊 **Dataset Sources**")

# Data for the pie chart
labels = ['MMSD 2.0 (Qin, et al., 2023)', 'Web Scraping', 'Synthetic Data']
sizes = [25000, 1500, 1500]
colors = ['#ff9999', '#66b3ff', '#99ff99']
explode = (0.1, 0, 0)  # Explode the largest slice

# Create the pie chart with enhanced aesthetics
fig, ax = plt.subplots(figsize=(3, 1.5))  # Further reduced figure size
ax.pie(sizes, explode=explode, labels=None, colors=colors, autopct='%1.1f%%', 
       shadow=True, startangle=140, textprops={'fontsize': 4})  # Labels removed from pie chart
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

# Add a legend on the right side
ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=5)

# Display the pie chart
st.pyplot(fig)

# Additional text info
st.write("""
- **25,000** 🗂️: Translated from MMSD 2.0 by **(Qin, et al., 2023)** using ChatGPT.
- **1,500** 🌐: Web-scraped from social media platforms.
- **1,500** 🛠️: Synthetic data created by the authors.
""")

# Dataset partitioning with clear icons
st.subheader("📂 **Dataset Partitioning**")
st.write("""
The dataset is split into three parts for effective model training and evaluation:
- **70% (19,600)** 🏋️‍♂️: Used for **training** the model.
- **10% (2,800)** 🔍: Reserved for **validation**.
- **20% (5,600)** 🧪: Set aside for **testing** the system.
""")

# Before and during implementation with step icons
st.subheader("🔨 **Before and During System Implementation**")
st.write("""
1. 🌍 **Translation**: 25k data from MMSD 2.0 is translated into Code-Switched Tagalog-English using OpenAI GPT.
2. 🕵️‍♀️ **Web Scraping**: 1.5k Code-Switched data collected from social media.
3. 🧪 **Data Generation**: 1.5k synthetic data created by the authors.
4. 📐 **Partitioning**: The dataset is split into training, validation, and testing sets.
5. 🤖 **Model Training**: 70% of the dataset is used to train the sarcasm detection system.
6. 🏗️ **System Development**: A system is being developed to classify sarcastic and non-sarcastic image-text posts.
7. 🧪 **Validation**: The system is validated using the 10% validation set after training.
8. 🔄 **Testing & Refinement**: Ongoing testing to improve and fine-tune the model.
""")

# Post-implementation section with testing icons
st.subheader("🚀 **After System Implementation**")
st.write("""
1. 🧪 **Testing**: The 20% testing dataset is input into the system for final evaluation.
2. 📊 **Data Gathering**: Researchers collect output data to calculate precision, recall, f-measure, and accuracy metrics.
3. ⚖️ **Model Comparison**: The performance of the BERT-based model (Pan, Lin, Fu, Qi, & Wang, 2020) is compared with the new RoBERTa and ResNet models.
4. 🧠 **Expert Validation**: Results are validated by NLP and linguistic experts for accuracy.
""")
