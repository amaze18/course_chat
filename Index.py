import nest_asyncio
nest_asyncio.apply()
# from github import Github
import boto3
import os
import openai
openai.api_key=os.environ['SECRET_TOKEN']
from llama_index.core.directory_reader import SimpleDirectoryReader
from llama_index.extractors.entity_extractor import EntityExtractor
from llama_index.parsers.sentence_splitter import SentenceSplitter
from llama_index.ingestion.pipeline import IngestionPipeline
from llama_index.core.service_context import ServiceContext
from llama_index.core.vector_store_index import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.storage_context import StorageContext, load_index_from_storage
from llama_index.memory.chat_memory import ChatMemoryBuffer
from llama_index.embeddings.openai_embedding import OpenAIEmbedding
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

s3 = boto3.client('s3')
# github_token = os.environ['GITHUB_TOKEN']
# github_repo = os.environ['GITHUB_REPO']

session = boto3.Session(
    aws_access_key_id=os.environ['ACCESS_ID'],
    aws_secret_access_key=os.environ['ACCESS_KEY'],
)
s3 = session.resource('s3')

bucket_name = 'coursechat'  # Replace with your actual S3 bucket name

# Option 1: Using list_objects with SortOrder and Prefix (if applicable)
# This approach works if you want to download the latest object from a specific prefix

# Specify a prefix to narrow down objects (optional)
prefix = '/'  # Replace with a prefix if you want to filter by folder (optional)

response = s3.list_objects(Bucket=bucket_name, Prefix=prefix, SortOrder='Descending')

# Check if there are any objects
if 'Contents' not in response:
    print("No objects found in the bucket or specified prefix.")
else:
    # Get the key of the first object (latest upload based on sort order)
    latest_object_key = response['Contents'][0]['Key']

    # Specify the desired download path
    download_path = '/home/ubuntu'  # Replace with your desired path

    local_filename = f"{download_path}/{latest_object_key}"
    s3.download_file(bucket_name, latest_object_key, local_filename)

def indexgenerator(indexPath, documentsPath):

    # check if storage already exists
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



indexPath=r"/"
documentsPath=f"/home/ubuntu"
indexgenerator(indexPath,documentsPath)

# def push_to_github(file_path):
#     # Authenticate with GitHub
#     g = Github(github_token)
#     repo = g.get_repo(github_repo)

#     # Push the file to the repository
#     with open(file_path, 'r') as file:
#         content = file.read()
#         repo.create_file(file_path, "Commit message", content)
