from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import os
from dotenv import load_dotenv
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
 
# Load environment variables from .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)  # Explicitly loading the .env
# Check if the key is loaded
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables!")

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = openai_api_key

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

file_path = ('../data/customers-100.csv')
loader = CSVLoader(file_path=file_path)
docs = loader.load_and_split()
print(f"Loaded {len(docs)} docs")
embeddings = OpenAIEmbeddings()
index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query(" ")))
vector_store = FAISS(
    embedding_function=OpenAIEmbeddings(),
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store.add_documents(documents=docs)

retriever = vector_store.as_retriever()

# Set up system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    
])

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

answer= rag_chain.invoke({"input": "How many customers living on Vietnam are there?"})
print(answer)

print(answer['answer'])