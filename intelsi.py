import streamlit as st
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
from datetime import datetime
import os
from github import Github

openai_api_key = st.secrets["OPENAI_API_KEY"]
github_client = Github(st.secrets["GITHUB_TOKEN"])
repo = github_client.get_repo(st.secrets["GITHUB_REPO"])

APPROVED_EMAILS = ["james@shmooze.io", "james.vineburgh@magellaneducation.co"]

def is_email_approved(email):
    return email in APPROVED_EMAILS

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

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    content_dir = "content"
    os.makedirs(content_dir, exist_ok=True)
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.txt")
    file_path = os.path.join(content_dir, filename)
    with open(file_path, 'a') as f:
        f.write(f"User: {input_text}\n")
        f.write(f"Chatbot response: {response.response}\n")
    with open(file_path, 'rb') as f:
        contents = f.read()
        repo.create_file(f"content/{filename}", f"Add chat file {filename}", contents)
    return response.response

def main():
    st.title("Document Submission and Management App")
    docs_directory_path = "docs"

    with st.container():
        st.header("User Submission")
        email = st.text_input("Enter your email address")
        
        if email:
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
                        repo.create_file(f"docs/{uploaded_file.name}", "Upload document", file_content)
                        st.success("File uploaded successfully to GitHub.")
                        construct_index(docs_directory_path)
            else:
                st.warning("Your email address is not approved for submission.")
        else:
            st.info("Please enter your email address to proceed.")

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
