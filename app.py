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


from pathlib import Path
from llama_index.agent.openai.base import OpenAIAgent
from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import SentenceSplitter
from tqdm.notebook import tqdm
import pickle
from llama_index.core.objects import (
    ObjectIndex,
    ObjectRetriever,
)
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import SubQuestionQueryEngine

from llama_index.readers.file import UnstructuredReader



from llama_index.core import Settings
from llama_index.core.schema import QueryBundle
from llama_index.core.agent import ReActAgent



# Define Embed+LLM globally
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

llm=OpenAI(model="gpt-3.5-turbo-0125", temperature=0)

st.set_page_config(page_title="VidyaRANG: Learning Made Easy",page_icon="/home/ubuntu/vidyarang.jpg",layout="centered")
DEFAULT_CONTEXT_PROMPT_TEMPLATE = """
  You're VidyaRANG. An AI assistant developed by members of AIGurukul to help students learn their course material via convertsations.
  The following is a friendly conversation between a user and an AI assistant for answering questions related to query.
  The assistant is talkative and provides lots of specific details from its context only.
  Here are the relevant documents for the context:

  {context_str}

  Instruction: Based on the above context, provide a crisp answer IN THE USER'S LANGUAGE with logical formation of paragraphs for the user question below.
  Strict Instruction: Answer "I don't know." if information is not present in context. Also, decline to answer questions that are not related to context."
  """
condense_prompt = (
  "Given the following conversation between a user and an AI assistant and a follow up question from user,"
  "rephrase the follow up question to be a standalone question.\n"
  "Chat History:\n"
  "{chat_history}"
  "\nFollow Up Input: {question}"
  "\nStandalone question:")

s3_bucket_name="coursechat"
access_key = os.environ.get('ACCESS_ID')
secret_key = os.environ.get('ACCESS_KEY')
auth_token = os.environ.get('auth_token')
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
st.title("VidyaRANG: Learning Made Easy")
#st.cache_data.clear()

REGION = 'us-east-1'
BUCKET_NAME = 'access-coursechat' 
s3c = boto3.client(
        's3', 
        region_name = REGION,aws_access_key_id=AWS_ACCESS_KEY ,aws_secret_access_key=AWS_SECRET_KEY
    )
def update_users():
    # Step 1: Access Google Sheet
    # Define the scope
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

    # Add credentials to the account
    creds = ServiceAccountCredentials.from_json_keyfile_name('trans-shuttle-418411-c40767daa58c.json', scope)

    # Authorize the client
    client = gspread.authorize(creds)

    # Open the Google Sheet by its URL
    url = 'https://docs.google.com/spreadsheets/d/1iDDl36tUIhTRWN4P2aCA7FUuBKXzuKHndmK4mtn3DOA/edit?resourcekey#gid=1112975212'
    worksheet = client.open_by_url(url).sheet1

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


def download_from_s3(course_name, download_path = '/home/ubuntu/documentsPath', bucket_name = 'coursechat'):
    
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


def nodesgenerator(documentsPath):

    entity_extractor = EntityExtractor(prediction_threshold=0.2,label_entities=False, device="cpu") # set device to "cuda" if gpu exists
    
    node_parser = SentenceSplitter(chunk_overlap=102,chunk_size=1024)
    
    transformations = [node_parser, entity_extractor]
    
    documents = SimpleDirectoryReader(input_dir=documentsPath).load_data()
    
    pipeline = IngestionPipeline(transformations=transformations)
    
    nodes = pipeline.run(documents=documents)
    
    return nodes
    
    
def build_agent_per_doc(nodes, indexPath, sindexPath, course_name):
     # check if storage already exists
    if not os.path.exists(indexPath):
        print("Not existing")
        service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
        vindex = VectorStoreIndex(nodes, service_context=service_context)

        # store it for later
        vindex.storage_context.persist(indexPath)
    else:
        #load existing index
        print("Existing")
        storage_context = StorageContext.from_defaults(persist_dir=indexPath)
        vindex = load_index_from_storage(storage_context,service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model))

    # build summary index
    summary_index = SummaryIndex(nodes)

    # define query engines
    vector_query_engine = vindex.as_query_engine(llm=llm)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize", llm=llm
    )

    # extract a summary
    if not os.path.exists(sindexPath):
        Path(sindexPath).parent.mkdir(parents=True, exist_ok=True)
        summary = str(
            summary_query_engine.aquery(
                "Extract a concise 1-2 line summary of this document"
            )
        )
        pickle.dump(summary, open(sindexPath, "wb"))
    else:
        summary = pickle.load(open(sindexPath, "rb"))

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{course_name}",
                description=f"Useful for questions related to specific facts",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name=f"summary_tool_{course_name}",
                description=f"Useful for summarization questions",
            ),
        ),
    ]

    # build agent
    function_llm = OpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=OpenAI(model="gpt-3.5-turbo-0125", temperature=0),
        verbose=True,
        system_prompt = DEFAULT_CONTEXT_PROMPT_TEMPLATE,
        prefix_messages = condense_prompt,
    )

    return agent, summary


def build_agents(documentsPath, indexPath, sindexPath, course_name):
    # Build agents dictionary
    agents_dict = {}
    extra_info_dict = {}

    # # this is for the baseline
    # all_nodes = []

    for filename in os.listdir(documentsPath):
        
        nodes = nodesgenerator(f"{documentsPath}/{filename}")
        
        agent, summary = build_agent_per_doc(nodes, indexPath, sindexPath, course_name)

        agents_dict[filename] = agent
            
        extra_info_dict[filename] = {"summary": summary, "nodes": nodes}

    return agents_dict, extra_info_dict



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


def get_indexed_course_list():
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
        
        
        
       
        
        
def course_chat(option,username=username):

    indexPath=f"/home/ubuntu/Indices/{option}"  
    sindexPath=f"/home/ubuntu/SIndices/{option}"
    documentsPath=f"/home/ubuntu/documentsPath/"
    #storage_context = StorageContext.from_defaults(persist_dir=indexPath)
    #index = load_index_from_storage(storage_context,service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model))
    if username == "GB123" or username == "Koosh0610" or username == "anupam.aiml@gmail.com" or username == "Helloworld" :
        llm = OpenAI(model="gpt-4-1106-preview")
        topk = 5
    else:
        llm = llm
        topk= 2  
    
    agents_dict, extra_info_dict = build_agents(documentsPath, indexPath, sindexPath, option)
    
    
    # define tool for each document agent
    all_tools = []
    for documents, agent in agents_dict.items():
        summary = extra_info_dict[documents]["summary"]
        doc_tool = QueryEngineTool(
            query_engine=agent,
            metadata=ToolMetadata(
                name=f"tool_{documents}",
                description=summary,
            ),
        )
        all_tools.append(doc_tool)




    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
    )
    vector_node_retriever = obj_index.as_node_retriever(
        similarity_top_k=topk,
    )


    # define a custom object retriever that adds in a query planning tool
    class CustomObjectRetriever(ObjectRetriever):
        def __init__(
            self,
            retriever,
            object_node_mapping,
            node_postprocessors=None,
            llm=None,
        ):
            self._retriever = retriever
            self._object_node_mapping = object_node_mapping
            self._llm = llm
            self._node_postprocessors = node_postprocessors

        def retrieve(self, query_bundle):
            if isinstance(query_bundle, str):
                query_bundle = QueryBundle(query_str=query_bundle)

            nodes = self._retriever.retrieve(query_bundle)
            for processor in self._node_postprocessors:
                nodes = processor.postprocess_nodes(
                    nodes, query_bundle=query_bundle
                )
            tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

            sub_question_engine = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=tools, llm=self._llm
            )
            sub_question_description = f"""\
    Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this \
    tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
    """
            sub_question_tool = QueryEngineTool(
                query_engine=sub_question_engine,
                metadata=ToolMetadata(
                    name="compare_tool", description=sub_question_description
                ),
            )

            return tools + [sub_question_tool]

    # wrap it with ObjectRetriever to return objects
    custom_obj_retriever = CustomObjectRetriever(
        vector_node_retriever,
        obj_index.object_node_mapping,
        node_postprocessors=[CohereRerank(top_n=5)],
        llm=llm,
    )    
    # get top document from top_agent
    top_agent = OpenAIAgent.from_tools(
        tool_retriever=custom_obj_retriever,
        system_prompt=""" \
    You are an agent designed to answer queries about the documentation.
    Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

    """,
        llm=llm,
        verbose=True,
    )
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [{"role": "assistant", "content": "Ask me a question from the course you have selected!!"}]
    if "message_history" not in st.session_state.keys():
        st.session_state.message_history=[ChatMessage(role=MessageRole.ASSISTANT,content="Ask me a question form the course you have selected"),]
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
            #doubt-2: 
                all_nodes = [n for extra_info in extra_info_dict.values() for n in extra_info["nodes"]]
                start = time.time()
                response = top_agent.query(str(prompt))
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


def upload_files(course_name, download_path = f"/home/ubuntu/documentsPath"):
    
    st.header("Upload Files")

    # Type box to get input for course name
    #course_name = st.text_input("Enter course name:")

    # File uploader
    uploaded_file_list = st.file_uploader("Choose your files (Uploader of the file confirms that the content being uploaded is original and does not violate any copyrights, granting permission for its distribution.)", type=["pdf", "txt", "csv","doc","docx","xls","xlsx"], accept_multiple_files=True)

    if st.button("Upload your files") and uploaded_file_list:
        for uploaded_file in uploaded_file_list:
            upload_to_s3(course_name, uploaded_file)
        download_from_s3(course_name)
        sindexPath=f"/home/ubuntu/SIndices/{course_name}"
        indexPath=f"/home/ubuntu/Indices/{course_name}"
        documents=download_path
        with st.spinner('Onboarding course:'):
            agents_dict, extra_info_dict = build_agents(documents, indexPath, sindexPath, course_name) 
            shutil.rmtree(documents)
            push_directory_to_github(indexPath, repo_owner, repo_name, token,branch_name,course_name)
            push_directory_to_github(sindexPath, repo_owner, repo_name, token,branch_name,course_name)
        st.success('You are all set to chat with your course!')
        st.write("Select action: Course Chat")


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

def chat_reset():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question from the course you have selected!!"}]
    #if 'chat_engine' in st.session_state:
        #del st.session_state.chat_engine

## MAIN FUNCTION ##
def main():
    
    # Sidebar
    # action = st.sidebar.selectbox("Select Action", ["Create New Course", "Upload Files"])
    if LOGGED_IN == True:
      #username= __login__obj.get_username()
      #st.header(f"Hi {username}! Welcome back")
      action=st.selectbox("Select Action",["Create New course","Update a existing course","Course chat"])
      if action == "Create New course":
           course_name = st.text_input("Course name:")
           upload_files(course_name)
      elif action == "Update a existing course":
           course_name = st.text_input("Course name:")
           upload_files(course_name)
      elif action == "Course chat":
           option= st.selectbox("Select course",tuple(get_indexed_course_list()),on_change=chat_reset)
           #chat_reset(option)
           course_chat(option)
      #st.footer("The user is solely responsible for uploading any content that may be subject to copyright.")
      st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #000000;
            padding: 10px 0;
            text-align: center;
            z-index: 9999;
        }
        </style>
        <div class="footer">
            The user is solely responsible for uploading any content that may be subject to copyright.
        </div>
        """,
        unsafe_allow_html=True
     )
# Push the directory to GitHub
if __name__ == "__main__":
    main()
