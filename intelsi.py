import streamlit as st
import pandas as pd
import os
from github import Github
import base64
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import openai
import datetime

# Authentication (Dummy list of approved emails for now)
APPROVED_EMAILS = ["user1@example.com", "user2@example.com"]

# Initialize GitHub client (assuming GitHub token is set in Streamlit secrets)
github_token = st.secrets["GITHUB_TOKEN"]
github_client = Github(github_token)
repo = github_client.get_repo(st.secrets["GITHUB_REPO"])

# Function to check if an email is approved
def is_email_approved(email):
    return email in APPROVED_EMAILS

# Function to upload file to GitHub
def upload_file_to_github(file_content, path, message):
    try:
        content_base64 = base64.b64encode(file_content.encode())
        repo.create_file(path, message, content_base64.decode())
        return True
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return False

# Main app
def main():
    st.title("Document Submission and Management App")

    # User page for document submission
    with st.container():
        st.header("User Submission")
        email = st.text_input("Enter your email address")
        if is_email_approved(email):
            with st.form(key='user_form'):
                first_name = st.text_input("First Name")
                last_name = st.text_input("Last Name")
                company_name = st.text_input("Company Name")
                phone_number = st.text_input("Phone Number")
                opportunity_name = st.text_input("Opportunity Name")
                uploaded_file = st.file_uploader("Upload File", type=['pdf', 'docx'])
                submit_button = st.form_submit_button(label='Submit')

                if submit_button and uploaded_file is not None:
                    # Save file to temporary directory and upload to GitHub
                    file_path = os.path.join('temp_docs', uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Upload file to GitHub
                    if upload_file_to_github(file_path, f"docs/{uploaded_file.name}"):
                        st.success("File uploaded successfully to GitHub.")

        else:
            st.warning("Your email address is not approved for submission.")

    # Admin page with chat and query functionalities
    with st.container():
        st.header("Admin Page")
        admin_password = st.text_input("Enter Admin Password", type="password")
        if admin_password == st.secrets["ADMIN_PASSWORD"]:
            st.success("Authenticated as Admin.")

            # Admin functionalities for document analysis and querying
            st.subheader("Document Query and Analysis Interface")
            query = st.text_area("Enter your query or chat with the AI")
            if st.button("Process"):
                # Process the query using GPT-Index, LLMPredictor, or other methods
                # response = index.query(query)  # Example query processing
                # For now, using OpenAI directly as a placeholder
                response = openai.Completion.create(engine="text-davinci-003", prompt=query, max_tokens=150)
                response_text = response.choices[0].text.strip()
                st.write(response_text)

                # Save chat history to GitHub
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                chat_history_path = f"content/chat_history_{timestamp}.txt"
                chat_content = f"Query: {query}\nResponse: {response_text}\n"
                upload_file_to_github(chat_content, chat_history_path, "Save chat history")

        else:
            st.error("Incorrect password. Access denied.")

if __name__ == "__main__":
    main()
