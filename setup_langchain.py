from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import tiktoken
import os
import json

# Initialize the OpenAI client
embeddings = OpenAIEmbeddings()
processed_data = []
client = OpenAI()

# Initialize the text splitter
tokenizer = tiktoken.get_encoding('p50k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

# Load in a professor's reviews
loader = WebBaseLoader("https://www.ratemyprofessors.com/professor/1810500") # review["id"]
data = loader.load()
texts = text_splitter.split_documents(data)

# Insert data into Pinecone
vectorstore = PineconeVectorStore(index_name="rag", embedding=embeddings)
index_name = "rag"
namespace = "rmp"
vectorstore_from_texts = PineconeVectorStore.from_texts([f"Source: {t.metadata['source']}, Title: {t.metadata['title']} \n\nContent: {t.page_content}" for t in texts], embeddings, index_name=index_name, namespace=namespace)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to Pinecone index
pinecone_index = pc.Index("rag")

# Perform RAG
query = "What are the reviews for all professors?"
raw_query_embedding = client.embeddings.create(
    input=[query],
    model="text-embedding-3-small"
)
query_embedding = raw_query_embedding.data[0].embedding
top_matches = pinecone_index.query(vector=query_embedding, top_k=10, include_metadata=True, namespace="rmp")

# Get the list of retrieved texts
contexts = [item['metadata']['text'] for item in top_matches['matches']]

# Create the augmented query using the retrieved contexts
augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
primer = r"""Please take the data provided and format it exactly as follows (do not include ```json```):
{
    "reviews": [
        {
            "professor": str,
            "review": str,
            "subject": str,
            "stars": float
        }
    ]
}
For the subject, just include the department name. If there are multiple reviews for the same professor, make sure to combine them into one entry with just one review.
"""

# Make the request to the OpenAI model
res = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)

openai_answer = res.choices[0].message.content
print(json.loads(openai_answer))