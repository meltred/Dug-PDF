import dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

gemini_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

reviews = Chroma.from_documents(
    reviews, gemini_embedding, persist_directory=REVIEWS_CHROMA_PATH
)

