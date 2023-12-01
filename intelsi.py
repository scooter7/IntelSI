import streamlit as st
import os
from datetime import datetime
import base64
import pickle
from io import BytesIO
from github import Github, GithubException
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI

# Initialize GitHub client
github_client = Github(st.secrets["GITHUB_TOKEN"])
repo = github_client.get_repo("scooter7/IntelSI")

APPROVED_EMAILS = ["james@shmooze.io", "james.vineburgh@magellaneducation.co"]

def is_email_approved(email):
    return email in APPROVED_EMAILS

def upload_file_to_github(file_path, message, file_content):
    try:
        contents = repo.get_contents(file_path)
        repo.update_file(file_path, message, file_content, contents.sha)
    except GithubException:
        repo.create_file(file_path, message, file_content)

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    return GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

def serialize_index_to_github(index, github_path):
    byte_stream = BytesIO()
    pickle.dump(index, byte_stream)
    byte_stream.seek(0)
    encoded_content = base64.b64encode(byte_stream.read()).decode()
    upload_file_to_github(github_path, "Update index", encoded_content)

def load_index_from_github(github_path):
    try:
        contents = repo.get_contents(github_path)
        encoded_content = contents.content
        decoded_content = base64.b64decode(encoded_content)
        byte_stream = BytesIO(decoded_content)
        return pickle.load(byte_stream)
    except GithubException as e:
        st.error("Error loading index from GitHub: " + str(e))
        return None

def chatbot(input_text):
    index = load_index_from_github('index.pkl')
    if index:
        response = index.query(input_text, response_mode="compact")
        content_dir = "content"
        os.makedirs(content_dir, exist_ok=True)
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.txt")
        file_path = os.path.join(content_dir, filename)
        with open(file_path, 'a') as f:
            f.write(f"User: {input_text}\n")
            f.write(f"Chatbot response: {response.response}\n")
        upload_file_to_github(f"content/{filename}", "Add chat file", open(file_path, 'rb').read())
        return response.response
    return "Error: Index not loaded."

def main():
    st.title("Document Submission and Management App")
    docs_directory_path = "docs"

    with st.container():
        st.header("User Submission")
        email = st.text_input("Enter your email address")
        
        if email and is_email_approved(email):
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
                    file_path = f"docs/{uploaded_file.name}"
                    upload_file_to_github(file_path, "Upload document", file_content)
                    st.success("File uploaded successfully to GitHub.")
                    index = construct_index(docs_directory_path)
                    serialize_index_to_github(index, 'index.pkl')

    with st.container():
        st.header("Admin Page")
        if st.text_input("Enter Admin Password", type="password") == st.secrets["ADMIN_PASSWORD"]:
            chat_container = st.container()
            input_text = st.text_input("Enter your query:")
            if st.button("Send"):
                response = chatbot(input_text)
                with chat_container:
                    st.write(f"User: {input_text}")
                    st.write(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
