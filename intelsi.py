import streamlit as st
import pandas as pd
import os
from github import Github
import base64
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import openai
import datetime

APPROVED_EMAILS = ["james@shmooze.io", "james.vineburgh@magellaneducation.co"]

github_token = st.secrets["GITHUB_TOKEN"]
github_client = Github(github_token)
repo = github_client.get_repo(st.secrets["GITHUB_REPO"])

def is_email_approved(email):
    return email in APPROVED_EMAILS

import base64

def upload_file_to_github(file_content, path, message):
    try:
        contents = repo.get_contents(path)
        # If file exists, update it
        repo.update_file(path, message, base64.b64encode(file_content).decode(), contents.sha)
    except github.GithubException:
        # If file does not exist, create it
        repo.create_file(path, message, base64.b64encode(file_content).decode())

def main():
    st.title("Document Submission and Management App")

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
                    file_content = uploaded_file.read()
                    upload_file_to_github(file_content, f"docs/{uploaded_file.name}", "Upload document")
                    st.success("File uploaded successfully to GitHub.")
        else:
            st.warning("Your email address is not approved for submission.")

    with st.container():
        st.header("Admin Page")
        admin_password = st.text_input("Enter Admin Password", type="password")
        if admin_password == st.secrets["ADMIN_PASSWORD"]:
            st.success("Authenticated as Admin.")
            st.subheader("Document Query and Analysis Interface")
            query = st.text_area("Enter your query or chat with the AI")
            if st.button("Process"):
                response = openai.Completion.create(engine="text-davinci-003", prompt=query, max_tokens=150)
                response_text = response.choices[0].text.strip()
                st.write(response_text)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                chat_history_path = f"content/chat_history_{timestamp}.txt"
                chat_content = f"Query: {query}\nResponse: {response_text}\n"
                upload_file_to_github(chat_content.encode(), chat_history_path, "Save chat history")
        else:
            st.error("Incorrect password. Access denied.")

if __name__ == "__main__":
    main()
