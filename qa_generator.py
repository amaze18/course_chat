from milvus import default_server
default_server.start()

import os
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Function to get the content (only question) from the prompt to cache
def get_msg_func(data, **_):
    return data.get("messages")[-1].content

from gptcache import cache
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

onnx = Onnx()
cache_base = CacheBase('sqlite')
vector_base = VectorBase('milvus', host='127.0.0.1', port='19530', dimension=onnx.dimension)
data_manager = get_data_manager(cache_base, vector_base)
cache.init(
    pre_embedding_func=get_msg_func,
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)
cache.set_openai_key()

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

from langchain_community.document_loaders.pdf import PyPDFLoader

pdf_loader = PyPDFLoader("/home/saiganesh/Desktop/[Mark_Grinblatt,_Sheridan_Titman]_Financial_Markets.pdf")
docs_from_pdf = pdf_loader.load_and_split(text_splitter=splitter)
# print(docs_from_pdf[0])

from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_generation.base import QAGenerationChain
from gptcache.adapter.langchain_models import LangChainChat

chat = LangChainChat(chat=ChatOpenAI(temperature=0))
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=500, chunk_size=2000)
chain = QAGenerationChain.from_llm(chat, text_splitter=text_splitter)

# qa = chain.run(str(docs_from_pdf[0]))

# print(qa[0])

qa_pairs = []
for doc in docs_from_pdf:
    qa = chain.run(str(doc))
    qa_pairs.append(qa)

    
import csv
csv_file_path = "/home/saiganesh/Desktop/Dataset_finance.csv"
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Question", "Course"])  # Writing the header
    for qa in qa_pairs:
        for pair in qa:
            writer.writerow([pair["question"], "Finance"])

print(f"QA pairs have been saved to {csv_file_path}")


default_server.stop()
