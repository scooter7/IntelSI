import streamlit as st
import pandas as pd
from github import Github, GithubException
import openai
from datetime import datetime
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI

APPROVED_EMAILS = ["james@shmooze.io", "james.vineburgh@magellaneducation.co"]
github_token = st.secrets["GITHUB_TOKEN"]
github_client = Github(github_token)
repo = github_client.get_repo(st.secrets["GITHUB_REPO"])

def is_email_approved(email):
    return email in APPROVED_EMAILS

def upload_file_to_github(file_content, path, message):
    try:
        contents = repo.get_contents(path)
        repo.update_file(path, message, file_content, contents.sha)
    except GithubException:
        repo.create_file(path, message, file_content)

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.directory_path = directory_path
    index.save_to_disk('index.json')
    return index

def chatbot(input_text, first_name, email):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    prompt = f"{first_name} ({email}): {input_text}"
    response = index.query(prompt, response_mode="compact")
    content_dir = "content"
    filename = st.session_state.filename
    file_path = f"{content_dir}/{filename}"
    with open(file_path, 'a') as f:
        f.write(f"{first_name} ({email}): {input_text}\n")
        f.write(f"Chatbot response: {response.response}\n")
    with open(file_path, 'rb') as f:
        contents = f.read()
        repo.create_file(f"content/{filename}", f"Add chat file {filename}", contents)
    return response.response

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
            chat_container = st.container()
            form = st.form(key="my_form", clear_on_submit=True)
            if "first_send" in st.session_state and st.session_state.first_send:
                first_name = form.text_input("Enter your first name:", key="first_name")
                email = form.text_input("Enter your email address:", key="email")
                st.session_state.first_send = False
            else:
                first_name = st.session_state.first_name
                email = st.session_state.email
            input_text = form.text_input("Enter your message:")
            form_submit_button = form.form_submit_button(label="Send")
            if form_submit_button and input_text:
                filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.docx")
                st.session_state.filename = filename
                response = chatbot(input_text, first_name, email)
                with chat_container:
                    st.write(f"{first_name}: {input_text}")
                    st.write(f"Chatbot: {response}")
                st.session_state.first_name = first_name
                st.session_state.email = email
            form.empty()
        else:
            st.error("Incorrect password. Access denied.")

if __name__ == "__main__":
    main()
