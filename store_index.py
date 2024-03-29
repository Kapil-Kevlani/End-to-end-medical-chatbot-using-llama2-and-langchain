from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os


load_dotenv()

extracted_data = load_pdf("data/")

text_chunks = text_split(extracted_data)

embeddings = download_hugging_face_embeddings()
PINECONE_API_KEY = 'fd8264ea-1261-4159-94c1-416aa5c6778b'
PINECONE_API_ENV = 'gcp-starter'
import os
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

from pinecone import Pinecone, ServerlessSpec

# pc = Pinecone(api_key=PINECONE_API_KEY)

# index_name = "medical-chatbot"
# if index_name not in pc.list_indexes().names():
  
#   # Do something, such as create the index
#   pc.create_index(
#     name=index_name,
#     dimension=384,
#     metric='cosine',
#     spec=ServerlessSpec(
#       cloud="gcp-starter",
#       region="us-central1"
#     )
#   )
  
from pinecone import Pinecone

pc = Pinecone(api_key='fd8264ea-1261-4159-94c1-416aa5c6778b')
index = pc.Index("medical-bot")

docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name="medical-bot")