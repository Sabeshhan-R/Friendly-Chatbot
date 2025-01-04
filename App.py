import os
import json
import csv
import nltk
import datetime
import ssl
import streamlit as st
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

file_path = os.path.abspath(r"intents.json")
with open(file_path,"r") as file:
    intents = json.load(file)

vectorizer = joblib.load('Model/vectorizer.pkl')
clf = joblib.load('Model/chatbot_model.pkl')

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
counter = 0
def main():
    global counter
    st.set_page_config(
        page_title="Chatbot Using NLP",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        /* Main Background Color */
        .main {
            background-color: #e0f7fa;
        }
        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background-color: #1565c0;
            color: white;
        }
        .sidebar .sidebar-content h2 {
            color: white;
        }
        /* Typing Pad */
        .stTextArea textarea {
            font-size: 20px;
            color: #1a237e;
            background-color: white;
            border: 1px solid black;
            border-radius: 6px;
            padding-left:2px;
            padding-right:0px;
        }
        .stTextArea textarea:focus {
            outline: none;
            border: 2px solid #0d47a1;
        }
        /* Send Button */
        .stButton button {
            background-color: #0d47a1;
            color: white;
            border-radius: 5px;
            margin-top : 40px;
        }
        .stButton button:hover {
            background-color: #0b3d91;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    menu = ["Chat", "Conversation History", "About"]
    choice = st.sidebar.radio("Navigation", menu)

    if choice == "Chat":
        st.title("🤖 Chatbot Using NLP")
        st.markdown("<h3 style='color: #0d47a1;'>Start Chatting</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns([4, 1])

        with col1:
            if not os.path.exists('chat_log.csv'):
                with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

            counter += 1
            user_input = st.text_area("Type your message:", height=70, key=f"user_input_{counter}")
        with col2:
            submit_button = st.button("Send")

        if submit_button and user_input.strip():
            response = chatbot(user_input.strip())
            st.markdown(f"<p style='color: #1a237e;font-size : 30px;'><strong>Chatbot:</strong> {response}</p>", unsafe_allow_html=True)
            
            # Log the conversation
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input.strip(), response, timestamp])
            
            if response.lower() in ['goodbye', 'bye']:
                st.success("Thank you for chatting with me. Have a greatday!")
                st.stop()

    elif choice == "Conversation History":
    st.title("📜 Conversation History")
    st.markdown("<h3 style='color: #1565c0;'>Review Past Chats</h3>", unsafe_allow_html=True)

    try:
        # Check if the file exists
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                try:
                    # Skip the header row
                    next(csv_reader)
                    history_found = False  # Flag to check if there's any data
                    
                    # Process each row
                    for row in csv_reader:
                        if len(row) == 3:  # Ensure the row has the expected number of columns
                            st.markdown(f"<p style='color: #1a237e;'><strong>User:</strong> {row[0]}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #1565c0;'><strong>Chatbot:</strong> {row[1]}</p>", unsafe_allow_html=True)
                            st.caption(f"Timestamp: {row[2]}")
                            st.markdown("---")
                            history_found = True
                        else:
                            st.error("Corrupted row in chat log. Skipping.")
                    
                    if not history_found:
                        st.info("No conversation data found in the chat log.")
                except csv.Error as e:
                    st.error(f"Error reading the CSV file: {e}")
        else:
            st.info("No conversation history available yet.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


    elif choice == "About":
        st.title("ℹ️ About This Chatbot")
        st.markdown(
            """
            <h3 style='color: #0d47a1;'>Overview</h3>
            <p>This chatbot is built using Natural Language Processing (NLP) techniques:</p>
            <ul>
                <li><strong>Framework:</strong> Streamlit</li>
                <li><strong>NLP Libraries:</strong> NLTK, Scikit-learn</li>
                <li><strong>Vectorization:</strong> TF-IDF</li>
                <li><strong>Model:</strong> Logistic Regression</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )


if __name__ == '__main__':
    main()


            
