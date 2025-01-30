from src.helper import load_pdf,text_split,download_embedding
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
#from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain_pinecone import PineconeVectorStore
import pinecone
import os

load_dotenv()

PINECONE_API_KEY=os.environ.get('Pinecone_api')
PINECONE_ENV_KEY=os.environ.get('Pinecone_env')
os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
index_name="medical-chatbot"
filepath=r"embeddings\text_embedding"

pinecone_client=Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV_KEY)

extracted_data=load_pdf("data_source/")
text_chunks=text_split(extracted_data)
embedding=download_embedding(filepath)

doc_Search=PineconeVectorStore.from_texts([t.page_content for t in text_chunks],embedding,index_name=index_name)