from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from src.helper import load_embedding_model
from src.prompt import prompt_template
from flask import Flask,jsonify,Response,render_template
import os
import pinecone
from dotenv import load_dotenv

load_dotenv()

app=Flask(__name__)

PINECONE_API_KEY=os.environ.get('Pinecone_api')
PINECONE_ENV_KEY=os.environ.get('Pinecone_env')
os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
index_name="medical-chatbot"
filepath=r"embeddings\text_embedding"

pinecone_client=Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV_KEY)

Prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
chain_type_kwargs={"prompt":Prompt}

load_embedding=load_embedding_model(filepath)
doc_search=PineconeVectorStore.from_existing_index(index_name=index_name,embedding=load_embedding)

model_path = r"model/llama-2-7b-chat.Q2_K.gguf"

# Initialize the model with specific configurations
llm = CTransformers(model=model_path, model_type="llama", config={'max_new_tokens': 150,'temperature':5})
qa=RetrievalQA.from_chain_type(
   llm=llm,
   chain_type="stuff",
   retriever=doc_search.as_retriever(search_kwargs={'k':3}),
   return_source_documents=True,
   chain_type_kwargs=chain_type_kwargs
)

@app.route('/')
def index():
   return render_template("chat.html")

if __name__=="__main__":
   app.run()