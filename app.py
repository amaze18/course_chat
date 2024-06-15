
import shutil
import warnings
warnings.filterwarnings("ignore")
import time
from rouge import Rouge
import streamlit as st
from streamlit_lottie import st_lottie
import os
import io
import boto3
from botocore.exceptions import NoCredentialsError
import nest_asyncio
nest_asyncio.apply()
import csv
from github import Github
import boto3
import os
import openai

openai.api_key=os.environ.get('SECRET_TOKEN')

import requests
from youtube_transcript_api import YouTubeTranscriptApi



from llama_index.core import SimpleDirectoryReader
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.legacy.core.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.legacy import ServiceContext
from llama_index.legacy.llms import OpenAI
from llama_index.legacy.retrievers import BM25Retriever
from llama_index.legacy.retrievers import VectorIndexRetriever
from llama_index.legacy.retrievers import BaseRetriever
from llama_index.legacy.chat_engine import CondensePlusContextChatEngine
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.postprocessor import LongContextReorder
from llama_index.legacy.schema import MetadataMode
from llama_index.legacy.schema import QueryBundle
from llama_index.legacy import (StorageContext,load_index_from_storage)
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd




# Set the error handler to the custom function
st.set_option('client.showErrorDetails', False)

# Your Streamlit app code goes here

st.set_page_config(page_title="VidyaRANG: Learning Made Easy",page_icon="/home/ubuntu/vidyarang.jpg",layout="centered")
DEFAULT_CONTEXT_PROMPT_TEMPLATE_1 = """
  You're VidyaRANG. An AI assistant developed by members of AIGurukul to help students learn their course material via convertsations.
  The following is a friendly conversation between a user and an AI assistant for answering questions related to query.
  The assistant is talkative and provides lots of specific details from its context only.
  Here are the relevant documents for the context:

  {context_str}

  Instruction: Based on the above context, provide a crisp answer IN THE USER'S LANGUAGE with logical formation of paragraphs for the user question below.
  Strict Instruction: Answer "I don't know." if information is not present in context. Also, decline to answer questions that are not related to context."
  """
#Relaxed prompt with language identification, ans in form of bullet points or short paragraphs
DEFAULT_CONTEXT_PROMPT_TEMPLATE_2 = """
 You're an AI assistant to help students learn their course material via convertsations.
 The following is a friendly conversation between a user and an AI assistant for answering questions related to query.
 The assistant is talkative and provides lots of specific details in form of bullet points or short paras from the context.
 Here is the relevant context:


 {context_str}


 Instruction: Based on the above context, provide a detailed answer IN THE USER'S LANGUAGE with logical formation of paragraphs for the user question below.


 """


#Relaxed + Creative promt with better language identification, ans is a mix bullet points and short paragraphs
DEFAULT_CONTEXT_PROMPT_TEMPLATE_3 = """
 You're an AI assistant to help students learn their course material via convertsations.
 The following is a friendly conversation between a user and a Generative AI assistant for answering questions related to query.
 The assistant is descriptive and provides lots of specific details as a mix of bullet points and short paragraphs from the context.
 Here is the relevant context:


 {context_str}


 Instruction: Based on the above context, provide a detailed and creative answer in same LANGUAGE as of  USER'S question with logical formation of paragraphs for the user question below.


 """


DEFAULT_CONTEXT_PROMPT_TEMPLATE_4 = """
  You’re an AI assistant designed to help students learn their medical course material through conversations. The following is a professional conversation between a user and an AI assistant for answering medical-related questions. The assistant uses precise medical terminologies and provides detailed information in the form of bullet points or short paragraphs from the context. The assistant also emphasizes that the information provided is for educational purposes and advises consulting a licensed healthcare professional for medical advice.

Here is the relevant context:

{context_str}

Instruction: Based on the above context, provide a detailed answer IN THE USER’S LANGUAGE with logical formation of paragraphs for the user question below.

Feel free to provide your specific medical question, and I will respond with a detailed, medically accurate explanation.
"""


s3_bucket_name="coursechat"


access_key = ""
secret_key = ""
auth_token = ""


# Set your AWS credentials and region (replace with your own values)
AWS_ACCESS_KEY = access_key
AWS_SECRET_KEY =secret_key
S3_BUCKET_NAME = s3_bucket_name

token = ""
# Repository information
repo_owner = "amaze18"
repo_name = "course_chat"

# Branch name
branch_name = "index"
from streamlit_login_auth_ui.widgets import __login__
import pandas as pd
from io import StringIO

st.title("VidyaRANG: Learning Made Easy")
st.warning("Operational Timing:   9:30AM - 9:30PM IST")
#st.cache_data.clear()

REGION = 'us-east-1'
BUCKET_NAME = 'access-coursechat' 
s3c = boto3.client(
        's3', 
        region_name = REGION,aws_access_key_id=AWS_ACCESS_KEY ,aws_secret_access_key=AWS_SECRET_KEY
    )
def update_users():
    """
    Access a Google Sheet and read its data into a pandas DataFrame.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the Google Sheet.
    """

    # Step 1: Access Google Sheet
    # Define the scope
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

    # Add credentials to the account
    creds = ServiceAccountCredentials.from_json_keyfile_name('trans-shuttle-418411-294e9666109b.json', scope)

    # Authorize the client
    client = gspread.authorize(creds)

    # Open the Google Sheet by its URL
    url = 'https://docs.google.com/spreadsheets/d/1iDDl36tUIhTRWN4P2aCA7FUuBKXzuKHndmK4mtn3DOA/edit?resourcekey#gid=1112975212'
    worksheet = client.open_by_url(url).sheet1
#try:
#    worksheet = client.open_by_url(url).sheet1
#except gspread.exceptions.APIError as e:
#    print(e.response)

    # Step 2: Read Google Sheet into pandas DataFrame
    # Get all values from the worksheet
    data = worksheet.get_all_values()

    # Convert to DataFrame
    headers = data.pop(0)
    df = pd.DataFrame(data, columns=headers)

    allowed_emails_df = pd.read_csv('allowed_emails.csv')

    # Extract the existing emails from the DataFrame
    existing_emails = set(allowed_emails_df['email'])

    # Extract emails from the 'E-mail ID' column of the DataFrame
    emails_to_check = set(df['E-mail ID'])

    # Find emails that are not in allowed_emails.csv
    new_emails = emails_to_check - existing_emails

    # Append new emails to allowed_emails.csv
    with open('allowed_emails.csv', 'a') as file:
        for email in new_emails:
            file.write(f"{email}\n")
update_users()
#obj = s3c.get_object(Bucket= BUCKET_NAME , Key = "allowed_emails.csv")
#df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf8')
df = pd.read_csv("allowed_emails.csv", encoding='utf8')
#allowed_emails_csv_path = "allowed_emails.csv"
#allowed_emails_df = pd.read_csv('allowed_emails.csv')
allowed_emails_set = set(df['email'].str.lower())


__login__obj = __login__(auth_token = auth_token, 
                    company_name = "VidyaRANG",
                    width = 200, height = 250, 
                    logout_button_name = 'Logout', hide_menu_bool = False, 
                    hide_footer_bool = False, 
                    lottie_url = "https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json",
                    allowed_emails_set = allowed_emails_set)

LOGGED_IN , username = __login__obj.build_login_ui()
username_inp = __login__obj.get_username()


def create_s3_subfolder(course_name:str):
    """
    Create a new subfolder in an S3 bucket.

    Args:
        course_name (str): The name of the course.
        AWS_ACCESS_KEY (str): The AWS access key ID.
        AWS_SECRET_KEY (str): The AWS secret access key.
        S3_BUCKET_NAME (str): The name of the S3 bucket.

    Returns:
        None
    """

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    # Create a new subfolder under the specified bucket
    subfolder_path = f"{course_name}/"
    s3.put_object(Bucket=S3_BUCKET_NAME, Key=subfolder_path)

def upload_to_s3(course_name:str, file:object) -> None:
    """
    Upload a file to an S3 bucket.

    Args:
        file (file object): The file to upload.
        course_name (str): The name of the course.
        AWS_ACCESS_KEY (str): The AWS access key ID.
        AWS_SECRET_KEY (str): The AWS secret access key.
        S3_BUCKET_NAME (str): The name of the S3 bucket.

    Returns:
        None
    """
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


def download_from_s3(course_name:str, download_path:str = '/home/ubuntu', bucket_name:str = 'coursechat'):
    """
    Download files from an S3 bucket to a local directory.

    Args:
        bucket_name (str): The name of the S3 bucket.
        course_name (str): The name of the course.
        download_path (str): The path to the local directory to download the files.

    Returns:
        None
    """

    local_filename=f"{download_path}"
    s3=boto3.client('s3')
    os.makedirs(f"{download_path}/{course_name}",exist_ok = True)
    files = get_files_in_directory(bucket_name,f"{course_name}/")
    for f_name in files:
        s3.download_file(bucket_name, f_name , local_filename + "/" + f_name)

# Creates sub folder in the s3 bucket based on the userdefined coursename
def create_new_course(course_name:str) -> None:
    """
    Create a new course with the given name.

    Args:
        course_name (str): The name of the course.

    Returns:
        None
    """

    # Type box to get input for course name

    if st.button("Create Course"):
        # Create a new subfolder in S3
        create_s3_subfolder(course_name)

        # Store the course name as a variable
        st.success(f"Course '{course_name}' created successfully! Subfolder in S3 created.")
    



def indexgenerator(indexPath:str, documentsPath:str) -> VectorStoreIndex:
    """
    Check if the index exists, if not, create it.

    Args:
        indexPath (str): The path to the index.
        documentsPath (str): The path to the documents.

    Returns:
        VectorStoreIndex: The loaded or created index.
    """

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
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-0125", temperature=0),embed_model=embed_model)
        index = VectorStoreIndex(nodes, service_context=service_context)

        # store it for later
        index.storage_context.persist(indexPath)
    else:
        #load existing index
        print("Existing")
        storage_context = StorageContext.from_defaults(persist_dir=indexPath)
        index = load_index_from_storage(storage_context,service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0),embed_model=embed_model))

    return index

def push_directory_to_github(directory_path:str, repo_owner:str, repo_name:str, token:str, branch_name:str,course_name:str) -> None:
    """
    Upload a directory to a specific branch in a GitHub repository.

    Args:
        directory_path (str): The path to the directory to be uploaded.
        course_name (str): The name of the course.
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        token (str): The GitHub personal access token.
        branch_name (str): The name of the branch in the GitHub repository.

    Returns:
        None
    """

    # Authenticate to GitHub using token
    g = Github(token)
    print(token,repo_owner,repo_name)
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
        contents = repo.get_contents(f"Indices/{course_name}", ref=branch.commit.sha)
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
            repo.create_file(f"Indices/{dir_name}/{file_name}", f"Add {file_name}", content_utf8, branch=branch_name)


def get_indexed_course_list() -> list:
    """
    Get a list of index files from the GitHub repository.

    Args:
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        token (str): The GitHub personal access token.

    Returns:
        list: A list of index files.
    """
    g = Github(token)

    # Get the repository
    repo = g.get_user(repo_owner).get_repo(repo_name)
    branch = repo.get_branch("index")
    try:
       contents = repo.get_contents("Indices", ref=branch.commit.sha)
       print(contents)
       x=[]
       for content_file in contents:
           x.append(content_file.path[8:])
       return x
    except:
        return []
    
def course_chat(option:str,username:str=username) -> None:
    """
    Initiate a chat session for a selected course.

    Args:
        option (str): The selected course.
        username (str): The username of the user initiating the chat.
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        token (str): The GitHub personal access token.
        branch_name (str): The name of the branch in the GitHub repository.

    Returns:
        None
    """
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    indexPath=f"/home/ubuntu/Indices/{option}"  
    storage_context = StorageContext.from_defaults(persist_dir=indexPath)
    index = load_index_from_storage(storage_context,service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0),embed_model=embed_model))
    if username == "GB123" or username == "Koosh0610" or username == "anupam.aiml@gmail.com" or username == "Helloworld" :
        llm = OpenAI(model="gpt-4-1106-preview")
        topk = 10
        
    else:
        llm = OpenAI(model="gpt-3.5-turbo-0125")
        topk= 2
    
    vector_retriever = VectorIndexRetriever(index=index,similarity_top_k=topk)
  
    bm25_flag = True
    try:
        bm25_retriever = BM25Retriever.from_defaults(index=index,similarity_top_k=topk)
    except:
        source_nodes = index.docstore.docs.values()
        nodes = list(source_nodes)
        bm25_flag = False
      
    postprocessor = LongContextReorder()
    rouge = Rouge()
    class HybridRetriever(BaseRetriever):
        def __init__(self,vector_retriever, bm25_retriever):
            self.vector_retriever = vector_retriever
            self.bm25_retriever = bm25_retriever
            super().__init__()

        def _retrieve(self, query, **kwargs):
            bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
            vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
            all_nodes = bm25_nodes + vector_nodes
            query = str(query)
            all_nodes = postprocessor.postprocess_nodes(nodes=all_nodes,query_bundle=QueryBundle(query_str=query.lower()))
            return all_nodes[0:topk]

    if bm25_flag:
        hybrid_retriever=HybridRetriever(vector_retriever,bm25_retriever)
    else:
        hybrid_retriever=vector_retriever
    
    service_context = ServiceContext.from_defaults(llm=llm)
    query_engine=RetrieverQueryEngine.from_args(retriever=hybrid_retriever,service_context=service_context,verbose=True)
    prompt=st.selectbox("Select Action",["Restrictive Prompt","Relaxed Prompt","Creative Prompt","Medical Prompt"])
    if prompt=="Restrictive Prompt":
       default_prompt = DEFAULT_CONTEXT_PROMPT_TEMPLATE_1
    elif prompt=="Relaxed Prompt":
       default_prompt = DEFAULT_CONTEXT_PROMPT_TEMPLATE_2
    elif prompt=="Medical Prompt":
       default_prompt = DEFAULT_CONTEXT_PROMPT_TEMPLATE_4
    else:
       default_prompt = DEFAULT_CONTEXT_PROMPT_TEMPLATE_3 
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [{"role": "assistant", "content": "Ask me a question from the course you have selected!!"}]
    if "message_history" not in st.session_state.keys():
        st.session_state.message_history=[ChatMessage(role=MessageRole.ASSISTANT,content="Ask me a questioin form the course you have selected"),]
    #if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        #st.session_state.chat_engine = CondensePlusContextChatEngine.from_defaults(query_engine,context_prompt=DEFAULT_CONTEXT_PROMPT_TEMPLATE,condense_prompt=condense_prompt,chat_history=st.session_state.message_history)
    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": str(prompt)})
    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                all_nodes  = hybrid_retriever.retrieve(str(prompt))
                start = time.time()
                response = CondensePlusContextChatEngine.from_defaults(query_engine,context_prompt=default_prompt,condense_prompt=prompt,chat_history=st.session_state.message_history).chat(str(prompt))
                end = time.time()
                st.write(response.response)
                context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in all_nodes])
                scores=rouge.get_scores(response.response,context_str)
                try:
                    df = pd.read_csv(f'logs/{option}.csv')
                    new_row = {'Question': str(prompt), 'Answer': response.response,'Unigram_Recall' : scores[0]["rouge-1"]["r"],'Unigram_Precision' : scores[0]["rouge-1"]["p"],'Bigram_Recall' : scores[0]["rouge-2"]["r"],'Bigram_Precision' : scores[0]["rouge-2"]["r"],"Time" : end-start}
                    df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
                    df.to_csv(f'logs/{option}.csv', index=False)
                    bucket = 'coursechat' # already created on S3
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer)
                    s3_resource= boto3.resource('s3',aws_access_key_id=os.environ["ACCESS_ID"],aws_secret_access_key=os.environ["ACCESS_KEY"])
                    s3_resource.Object(bucket, option+'_course_logs.csv').put(Body=csv_buffer.getvalue())
                    st.session_state.message_history.append(ChatMessage(role=MessageRole.ASSISTANT,content=str(response.response)),)
                except:
                    df = pd.DataFrame(columns=['Question','Answer','Unigram_Recall','Unigram_Precision','Bigram_Recall','Bigram_Precision','Time'])
                    new_row = {'Question': str(prompt), 'Answer': response.response,'Unigram_Recall' : scores[0]["rouge-1"]["r"],'Unigram_Precision' : scores[0]["rouge-1"]["p"],'Bigram_Recall' : scores[0]["rouge-2"]["r"],'Bigram_Precision' : scores[0]["rouge-2"]["r"],"Time" : end-start}
                    df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
                    df.to_csv(f'logs/{option}.csv', index=False)
                    bucket = 'coursechat' # already created on S3
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer)
                    s3_resource= boto3.resource('s3',aws_access_key_id=os.environ["ACCESS_ID"],aws_secret_access_key=os.environ["ACCESS_KEY"])
                    s3_resource.Object(bucket, f'{option}_course_logs.csv').put(Body=csv_buffer.getvalue())
                    st.session_state.message_history.append(ChatMessage(role=MessageRole.ASSISTANT,content=str(response.response)),)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history


def upload_files(course_name:str, download_path:str = '/home/ubuntu') -> None:
    """
    Upload files, process them, and initiate course chat.

    Args:
        course_name (str): The name of the course.
        download_path (str): The path to download the files.
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        token (str): The GitHub personal access token.
        branch_name (str): The name of the branch in the GitHub repository.

    Returns:
        None
    """
    
    st.header("Upload Files")

    # Type box to get input for course name
    #course_name = st.text_input("Enter course name:")

    # File uploader
    uploaded_file_list = st.file_uploader("Choose your files (Uploader of the file confirms that the content being uploaded is original and does not violate any copyrights, granting permission for its distribution.)", type=["pdf", "txt", "csv","doc","docx","xls","xlsx"], accept_multiple_files=True)

    if st.button("Upload your files") and uploaded_file_list:
        for uploaded_file in uploaded_file_list:
            upload_to_s3(course_name, uploaded_file)
        download_from_s3(course_name)

        indexPath=f"{download_path}/Indices/{course_name}"
        documents=f"{download_path}/{course_name}"
        with st.spinner('Onboarding course:'):
            indexgenerator(indexPath, documents)
            shutil.rmtree(documents)
            push_directory_to_github(indexPath, repo_owner, repo_name, token, branch_name, course_name)
        st.success('You are all set to chat with your course!')
        st.write("Select action: Course Chat")



def upload_files_yt(course_name:str, download_path:str = '/home/ubuntu') -> None:
    """
    Upload files, process them, and initiate course chat.

    Args:
        course_name (str): The name of the course.
        download_path (str): The path to download the files.
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        token (str): The GitHub personal access token.
        branch_name (str): The name of the branch in the GitHub repository.

    Returns:
        None
    """
    
    

    # Type box to get input for course name
    #course_name = st.text_input("Enter course name:")
    
    # File uploader
    #uploaded_file_list = st.file_uploader("Choose your files (Uploader of the file confirms that the content being uploaded is original and does not violate any copyrights, granting permission for its distribution.)", type=["pdf", "txt", "csv","doc","docx","xls","xlsx"], accept_multiple_files=True)
    #print("-----------",uploaded_file_list)
    if st.button("Upload your video") :
        uploaded_file = f"/home/ubuntu/ytscript/"
        #upload_to_s3(course_name, uploaded_file)
        #download_from_s3(course_name)

        indexPath=f"{download_path}/Indices/{course_name}"
        #documents=f"{download_path}/{course_name}"
        with st.spinner('Onboarding course:'):
            indexgenerator(indexPath, uploaded_file)
            os.remove(f"{uploaded_file}{course_name}.txt")
            push_directory_to_github(indexPath, repo_owner, repo_name, token, branch_name, course_name)
        st.success('You are all set to chat with your course!')
        st.write("Select action: Course Chat")


# Function to get video title using YouTube Data API
def get_video_title(api_key, video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=snippet"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch video details: {response.status_code}")
    data = response.json()
    title = data['items'][0]['snippet']['title']
    return title

# Function to get video transcript using youtube-transcript-api
def get_video_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # Join the transcript text in a readable format
        transcript = "\n".join([entry['text'] for entry in transcript_list])
        return transcript
    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {e}")

# Main function
def get_transcript(video_url,course_name):
    api_key = ""
    

    # Extract video ID from URL
    video_id = video_url.split("v=")[-1].split("&")[0]
    
    try:
        # Fetch video title
        title = get_video_title(api_key, video_id)
        print(f"Video Title: {title}\n")
        
        # Fetch video transcript
        transcript = get_video_transcript(video_id)
        print(f"Transcript:\n{transcript}")

        # Store title and transcript in a file
        with open(f"/home/ubuntu/ytscript/{course_name}.txt", "w", encoding="utf-8") as file:
            file.write(f"Video Title: {title}\n\nTranscript:\n{transcript}")

        print(f"\nTranscript saved to {course_name}.txt")
    except Exception as e:
        print(e)

def get_files_in_directory(bucket_name:str, directory:str) -> list:
    """
    List files in a directory within an S3 bucket.

    Args:
        s3: Boto3 S3 client instance.
        bucket_name (str): Name of the S3 bucket.
        directory (str): Directory within the bucket to list files from.

    Returns:
        list: A list of file names within the specified directory.
    """
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

def chat_reset():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question from the course you have selected!!"}]
    #if 'chat_engine' in st.session_state:
        #del st.session_state.chat_engine

# Function to check if the email exists in the specified CSV file
def check_user(email:str, file_path:str) -> bool:
    """
    Check if an email exists in a CSV file.

    Args:
        file_path (str): The path to the CSV file containing user data.
        email (str): The email address to check for existence.

    Returns:
        bool: True if the email exists in the CSV file, False otherwise.
              Returns False if there is an error reading the CSV file.
    """
    try:
        users_df = pd.read_csv(file_path)
        if email in users_df["email"].tolist():
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return False

# Function to get course names created by the logged-in teacher
def get_courses_for(email:str, file_path:str) -> list:
    """
    Retrieve courses associated with a given email from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing user data.
        email (str): The email address for which to retrieve courses.

    Returns:
        list: A list of courses associated with the specified email.
              Returns an empty list if there is an error reading the CSV file or if the email is not found.
    """
    try:
        df = pd.read_csv(file_path)
        courses = df[df['email'] == email]['course'].tolist()
        return courses
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return []
    
def check_blocked_email(email:str, csv_file:str) -> tuple[bool, str]:
    ''' Check access for a given email in a CSV file.

    Args:
        csv_file (str): The path to the CSV file containing user data.
        email (str): The email address to check for access.

    Returns:
        tuple: A tuple containing two elements:
            - A boolean indicating whether the email has access.
            - If the email has access, the type of access; otherwise, a message indicating the reason.
    '''
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row['email'] == email:
                if row['access']:
                    return True, row['access']
                else:
                    return False, "Access not blocked"
                
        return False, "Email not found"
    
# Example usage
email_to_check = username_inp
csv_file_path = "allowed_emails.csv"

blocked, reason = check_blocked_email(email_to_check, csv_file_path)

# Function to get the list of course names from a CSV file
def get_course_list_from_csv(file_path):
    """
    Get the list of courses from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of courses.
    """
    try:
        df = pd.read_csv(file_path)
        course_list = df["course"].tolist()
        return course_list
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return []

# Example usage:
course_list = get_course_list_from_csv("teachers.csv")

#function to check email is in instructor mode or not
def check_instructor_mode(csv_file_path, email):
    """
    Check if the email has instructor mode based on the CSV file.

    Args:
        csv_file_path (str): The path to the CSV file.
        email (str): The email to check.

    Returns:
        bool: True if the email has instructor mode, False otherwise.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)
        
        # Check if the email is present in the DataFrame
        if email in df["email"].tolist():
            # Get the mode corresponding to the email
            mode = df.loc[df["email"] == email, "mode"].iloc[0]
            # Check if mode is "instructor"
            if mode == "instructor":
                return True
            else:
                return False
        else:
            print(f"Email '{email}' not found in the CSV file.")
            return False
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False

#using Postgress DATABASE
import psycopg2

def check_instructor_DATABASE(email):
    """
    Check if the user with the given email has instructor access in the database.

    Args:
        email (str): The email of the user to check.
        dbname (str): The name of the database.
        user (str): The username for database authentication.
        password (str): The password for database authentication.
        host (str): The host address of the database.
        port (str): The port number of the database.

    Returns:
        bool: True if the user has instructor access, False otherwise.
    """
    # Database connection parameters
    dbname="app_login_db"
    user="cuser"
    password="123"
    host="localhost"
    port="5432"
    

    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        cur = conn.cursor()

        # Query to check restricted column for the given email
        query = """
            SELECT restricted
            FROM access_check
            WHERE email = %s
        """
	
        # Execute the query
        cur.execute(query, (email,))
        row = cur.fetchone()
	
        # Check the restricted column value
        if row:
            restricted = row[0]
            return restricted == 'instructor'
        else:
            return False

    except psycopg2.Error as e:
        print("-----------------Error:", e)
        return False

    finally:
        # Close cursor and connection
        if cur:
           cur.close()
        if conn:
           conn.close()

	
# Example usage
#using database
#instructor_access = check_instructor_DATABASE(username_inp)



teachers_csv_path = "instructor_access.csv"

instructor_access = check_instructor_mode(teachers_csv_path, username_inp)


def save_assignment_to_csv(course_name, users, file_path):
    """
    Save the assignment of users to a course to a CSV file.

    Args:
        course_name (str): The name of the course.
        users (list): The list of users assigned to the course.
        file_path (str): The path to the CSV file to save the assignment.

    Returns:
        None
    """

    try:
        # Create a new DataFrame with the course and users
        data = {'course': [course_name], 'users': [', '.join(users)]}
        df = pd.DataFrame(data)

        # If the file does not exist, write the header and data
        if not os.path.isfile(file_path):
            df.to_csv(file_path, index=False)
        else:
            # Otherwise, append the data without writing the header
            df.to_csv(file_path, mode='a', header=False, index=False)
        
        st.success(f"{users} can now access {course_name}.  Assigned successfully to {file_path}")
    except Exception as e:
        st.error(f"Error saving to CSV file: {e}")





# This function defines the main logic for a Streamlit application. 
#It handles user authentication, course management, and interaction based on user roles (instructor or learner)
def main(username=username):
    """
    Display actions based on user permissions and status.

    Args:
        LOGGED_IN (bool): Flag indicating whether the user is logged in.
        instructor_access (bool): Flag indicating whether the user has instructor access.
        username_inp (str): The username of the logged-in user.
        blocked (bool): Flag indicating whether the email is blocked.
        email_to_check (str): The email to check for blocking.
        reason (str): The reason for blocking the email.
        teachers_file_path (str): The file path to the teachers CSV file.
        student_file_path (str): The file path to the allowed emails CSV file.

    Returns:
        None
    """
    
    teachers_file_path = 'teachers.csv'
    student_file_path = 'allowed_emails.csv'
    #displays a message indicating that a particular email is blocked, along with the reason for blocking it
    if blocked:
        st.write(f"The email {email_to_check} is blocked. Reason: {reason}")
    else:
        st.warning("Operation Timing:   9:30AM - 9:30PM IST")
        if LOGGED_IN :
            #interface for instructors after they've successfully logged in 
            if instructor_access:
                st.success(f"You logged in as instructor: {username_inp}")
                courses = get_courses_for(username_inp, teachers_file_path)
                
                if courses:
                    
                    st.markdown(f"<span style='font-size: larger'><b>Courses created by you: </b></span>", unsafe_allow_html=True)
                    for course in courses:
                        st.markdown(f"- {course}")
                #presents a selection box for various actions like creating a new course, updating an existing one, or accessing a course chat
                action=st.selectbox("Select Action",["Create New course","Assign Course","Update a existing course","Course chat"])
                if action == "Create New course":
                    course_name = st.text_input("Course name:")
                    course_type = st.selectbox("Select course type", ["Public", "Private"])
                    
                    student_emails = None
                    if course_type == "Private":
                        student_emails = st.text_area("Enter student emails (comma-separated):")
                    
                    
                    st.title("Chat with Document or YouTube Video")

                    # Paths or URLs to the logo images
                    logo1_path = "/home/ubuntu/document.png"  # Placeholder URL for logo1
                    logo2_path = "/home/ubuntu/youtube.png"  # Placeholder URL for logo2

                    # Display the logos as buttons
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("Document", key="logo1"):
                            st.session_state.logo_clicked = "1"
                        st.image(logo1_path, caption="Upload Document")

                    with col2:
                        if st.button("YouTube Video", key="logo2"):
                            st.session_state.logo_clicked = "2"
                        st.image(logo2_path, caption="Paste YouTube Video link")

                    # Display text based on the button clicked
                    if "logo_clicked" in st.session_state:
                        if st.session_state.logo_clicked == "1":
                            upload_files(course_name)
                            
                        elif st.session_state.logo_clicked == "2":
                            st.header("Paste Youtube Link")
                            url = st.text_input("Paste link here (keep length of video under 6 minutes)")
                            get_transcript(url, course_name)
                            upload_files_yt(course_name)


                elif action == "Assign Course":
                    
                        
                    if courses:
                        selected_courses = st.multiselect("Select Courses", courses)
                        user_input = st.text_input("Enter Users (comma separated)")
            
                        if st.button("Assign"):
                            assigned_users = [user.strip() for user in user_input.split(',')]
                            #st.success(f"Users {', '.join(assigned_users)} assigned to courses '{', '.join(selected_courses)}' successfully!")
                            for course in selected_courses:
                                save_assignment_to_csv(course, assigned_users, "allowed_private_courses.csv")
    

                elif action == "Update a existing course":
                    course_name = st.text_input("Course name:")
                    upload_files(course_name)
                elif action == "Course chat":
                    option= st.selectbox("Select course",tuple(get_indexed_course_list()),on_change=chat_reset)
                    #chat_reset(option)
                    course_chat(option)
            #displays a success message with the learner's username and prompts them to select a course from a dropdown menu. 
            elif instructor_access == False:
                st.success(f"You logged in as learner: {username_inp}")
                option= st.selectbox("Select course",tuple(get_indexed_course_list()),on_change=chat_reset)
                course_chat(option)#ChatGPT
            else:
                st.error("Email or Username not Registered")

if __name__ == "__main__":
    main()
