import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')

# Step 1: Set up embeddings
embedding_model = CohereEmbeddings(model="embed-english-v3.0")

# Step 2: Define document URLs
urls = [
    "https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/?ref=dl-staging-website.ghost.io"
]

# Step 3: Load documents from URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Step 4: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

# Step 5: Create vector store using Chroma and the Cohere embeddings
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag",
    embedding=embedding_model,
)

# Set up retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 4})

# Step 6: Define question to retrieve documents
question = "What are the different kinds of agentic design patterns?"

# Step 7: Retrieve documents based on the question
docs = retriever.invoke(question)

# Step 8: Relevancy check using LLM
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
Give a binary score 'yes' or 'no'."""
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
])

retrieval_grader = grade_prompt | structured_llm_grader

# Filter relevant documents
docs_to_use = []
for doc in docs:
    res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    if res.binary_score == 'yes':
        docs_to_use.append(doc)

# Step 9: Generate answer
system = """You are an assistant for question-answering tasks. Answer the question based on retrieved documents."""
prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>")
])

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
rag_chain = prompt | llm | StrOutputParser()

def format_docs(docs):
    return "\n".join(f"<doc{i+1}>:\nTitle:{doc.metadata['title']}\nSource:{doc.metadata['source']}\nContent:{doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate(docs))

generation = rag_chain.invoke({"documents": format_docs(docs_to_use), "question": question})
print(generation)

# Step 10: Hallucination check
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in 'generation' answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

hallucination_grader = ChatPromptTemplate.from_messages([
    ("system", "Assess whether an LLM generation is grounded in / supported by a set of retrieved facts."),
    ("human", "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation>")
]) | structured_llm_grader

response = hallucination_grader.invoke({"documents": format_docs(docs_to_use), "generation": generation})
print(response)

# Step 11: Highlight segments used to generate answer
class HighlightDocuments(BaseModel):
    """Return the specific part of a document used for answering the question."""
    id: List[str] = Field(description="List of id of docs used to answers the question")
    title: List[str] = Field(description="List of titles used to answers the question")
    source: List[str] = Field(description="List of sources used to answers the question")
    segment: List[str] = Field(description="List of direct segments from used documents that answers the question")

llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
parser = PydanticOutputParser(pydantic_object=HighlightDocuments)

highlight_prompt = ChatPromptTemplate.from_messages([
    ("system", "Identify and extract the exact segments from the provided documents that directly correspond to the content used to generate the answer."),
    ("human", "Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>")
])

doc_lookup = highlight_prompt | llm | parser

lookup_response = doc_lookup.invoke({"documents": format_docs(docs_to_use), "question": question, "generation": generation})
for id, title, source, segment in zip(lookup_response.id, lookup_response.title, lookup_response.source, lookup_response.segment):
    print(f"ID: {id}\nTitle: {title}\nSource: {source}\nText Segment: {segment}\n")
