from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
import dotenv
import os

dotenv.load_dotenv()
key = os.getenv("GOOGLE_API_KEY")

def GenerateEmbedding(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
    return vector_index

