import streamlit as st
import os
import boto3
from botocore.exceptions import NoCredentialsError
import nest_asyncio
nest_asyncio.apply()

from github import Github
import boto3
import os
import openai
openai.api_key=os.environ['SECRET_TOKEN']

from llama_index.core import SimpleDirectoryReader
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding

s3_bucket_name="coursechat"
access_key = os.environ.get('ACCESS_ID')
secret_key = os.environ.get('ACCESS_KEY')
# Set your AWS credentials and region (replace with your own values)
AWS_ACCESS_KEY = access_key
AWS_SECRET_KEY =secret_key
S3_BUCKET_NAME = s3_bucket_name

token = os.environ['GITHUB_TOKEN']
# Repository information
repo_owner = "amaze18"
repo_name = "course_chat"

# Branch name
branch_name = "index"
from streamlit_login_auth_ui.widgets import __login__
import pandas as pd
from io import StringIO

st.cache_data.clear()

allowed_emails_csv_path = "allowed_emails.csv"
allowed_emails_df = pd.read_csv('allowed_emails.csv')
allowed_emails_set = set(allowed_emails_df['email'].str.lower())

__login__obj = __login__(auth_token = "courier_auth_token", 
                    company_name = "Shims",
                    width = 200, height = 250, 
                    logout_button_name = 'Logout', hide_menu_bool = False, 
                    hide_footer_bool = False, 
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

                    
LOGGED_IN = __login__obj.build_login_ui()
def create_s3_subfolder(course_name):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    # Create a new subfolder under the specified bucket
    subfolder_path = f"{course_name}/"
    s3.put_object(Bucket=S3_BUCKET_NAME, Key=subfolder_path)

def upload_to_s3(course_name, file):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    subfolder_path = f"{course_name}"
    s3_file_path = f"{subfolder_path}/{file.name}"
    try:
        s3.upload_fileobj(file, S3_BUCKET_NAME, s3_file_path)
        st.success(f"File '{file.name}' uploaded successfully to '{subfolder_path}'")
    except NoCredentialsError:
        st.error("AWS credentials not available. Please check your credentials.")


def download_from_s3(course_name, download_path = '/home/ubuntu', bucket_name = 'coursechat'):
    
    local_filename=f"{download_path}"
    s3=boto3.client('s3')
    os.makedirs(f"{download_path}/{course_name}",exist_ok = True)
    files = get_files_in_directory(bucket_name,f"{course_name}/")
    for f_name in files:
        s3.download_file(bucket_name, f_name , local_filename + "/" + f_name)

# Creates sub folder in the s3 bucket based on the userdefined coursename
def create_new_course(course_name):

    # Type box to get input for course name

    if st.button("Create Course"):
        # Create a new subfolder in S3
        create_s3_subfolder(course_name)

        # Store the course name as a variable
        st.success(f"Course '{course_name}' created successfully! Subfolder in S3 created.")
    



def indexgenerator(indexPath, documentsPath):
    # check if storage already exists
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    if not os.path.exists(indexPath):
        print("Not existing")
        # load the documents and create the index
        

        entity_extractor = EntityExtractor(prediction_threshold=0.2,label_entities=False, device="cpu") # set device to "cuda" if gpu exists
        node_parser = SentenceSplitter(chunk_overlap=102,chunk_size=1024)
        transformations = [node_parser, entity_extractor]

        documents = SimpleDirectoryReader(input_dir=documentsPath).load_data()

        pipeline = IngestionPipeline(transformations=transformations)
        nodes = pipeline.run(documents=documents)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0),embed_model=embed_model)
        index = VectorStoreIndex(nodes, service_context=service_context)

        # store it for later
        index.storage_context.persist(indexPath)
    else:
        #load existing index
        print("Existing")
        storage_context = StorageContext.from_defaults(persist_dir=indexPath)
        index = load_index_from_storage(storage_context,service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0),embed_model=embed_model))

    return index

def push_directory_to_github(directory_path, repo_owner, repo_name, token, branch_name,course_name):
    # Authenticate to GitHub using token
    g = Github(token)

    # Get the repository
    repo = g.get_user(repo_owner).get_repo(repo_name)

    # Create branch if not exists
    branches = repo.get_branches()
    branch_exists = False
    for b in branches:
        if b.name == branch_name:
            branch_exists = True
            break
    if not branch_exists:
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=repo.get_branch("main").commit.sha)

    # Create a folder in the repository with the directory name
    dir_name = os.path.basename(directory_path)
    branch = repo.get_branch("index")
    try:
        contents = repo.get_contents(course_name, ref=branch.commit.sha)
        for content_file in contents:
            repo.delete_file(content_file.path, f"Deleting {content_file.path}", content_file.sha, branch="index")
    except:
        print("deleting error")     
 

    # Push all files in the directory to the repository
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'rb') as file:  # Open the file in binary mode
            content = file.read()  # Read content as bytes
            # Convert bytes to UTF-8 encoded string
            content_utf8 = content.decode('utf-8', 'ignore')
            repo.create_file(f"{dir_name}/{file_name}", f"Add {file_name}", content_utf8, branch=branch_name)




def upload_files(course_name, download_path = '/home/ubuntu'):
    
    st.header("Upload Files")

    # Type box to get input for course name
    #course_name = st.text_input("Enter course name:")

    # File uploader
    uploaded_file_list = st.file_uploader("Choose a file", type=["pdf", "txt", "csv"], accept_multiple_files=True)

    if st.button("Upload File") and uploaded_file_list:
        # Upload file to S3
        for uploaded_file in uploaded_file_list:
            upload_to_s3(course_name, uploaded_file)
        download_from_s3(course_name)
        indexPath=f"{download_path}/index/{course_name}"
        documents=f"{download_path}/{course_name}"
        indexgenerator(indexPath, documents)
        push_directory_to_github(indexPath, repo_owner, repo_name, token,branch_name,course_name)


def get_files_in_directory(bucket_name, directory):
    s3 = boto3.client('s3')

    # Ensure the directory name ends with '/'
    if not directory.endswith('/'):
        directory += '/'

    # List objects in the specified directory
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=directory)
    
    # Extract file names from the response
    file_names = []
    if response['Contents']:
        for obj in response['Contents']:
            # Exclude directories and only consider files
            if not obj['Key'].endswith('/'):
                file_names.append(obj['Key'])

    return file_names


## MAIN FUNCTION ##
def main():
    st.title("Course Management App")
    
    # Sidebar
    # action = st.sidebar.selectbox("Select Action", ["Create New Course", "Upload Files"])
    if LOGGED_IN == True:
      #username= __login__obj.get_username()
      #st.header(f"Hi {username}! Welcome back")
      action=st.selectbox("Select Action",["Create New course","Update a existing course"])   
      course_name = st.text_input("Course name:")
      if action == "Create New course":
           create_new_course(course_name)
      elif action == "Update a existing course":
           upload_files(course_name)



# Push the directory to GitHub
if __name__ == "__main__":
    main()

