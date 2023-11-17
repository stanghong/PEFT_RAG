# First, ensure that these packages are installed in your environment.
# You can install or force-reinstall them using pip in your terminal.

# pip install --force-reinstall boto3 --quiet
# pip install langchain==0.0.305 --force-reinstall --quiet
# pip install pypdf==3.8.1 faiss-cpu==1.7.4 --force-reinstall --quiet
# pip install tiktoken==0.4.0 --force-reinstall --quiet
# pip install sqlalchemy==2.0.21 --force-reinstall --quiet
# %%
import boto3
import json
import os
import sys
from pypdf import PdfReader, PdfWriter
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Append the utils module path and import utilities
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww
from utils.TokenCounterHandler import TokenCounterHandler

# Set up the Bedrock client
bedrock_client = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
    runtime=True  # Default. Needed for invoke_model() from the data plane
)

# Initialize the token counter
token_counter = TokenCounterHandler()

# We will be using the Titan Embeddings Model to generate our Embeddings.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Create the LLM and Embeddings Model
llm = Bedrock(model_id="anthropic.claude-v2", 
              client=bedrock_client, 
              model_kwargs={'max_tokens_to_sample': 200}, 
              callbacks=[token_counter])

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock_client)
# %%
# Data Preparation
# In an actual script, you would download files or ensure they exist in the 'data' directory
# Here we assume the PDF files are already present in the './data/' directory

filenames = [
"Data_Scientist_Resume_0.pdf",
"Data_Scientist_Resume_1.pdf",
"Data_Scientist_Resume_3.pdf",
"Data_Scientist_Resume_2.pdf",
"Data_Scientist_Resume_6.pdf",
"Data_Scientist_Resume_7.pdf",
"Data_Scientist_Resume_5.pdf",
"Data_Scientist_Resume_4.pdf",
"Data_Scientist_Resume_10.pdf",
"Data_Scientist_Resume_9.pdf",
"Data_Scientist_Resume_8.pdf",
"Machine_Learning_Engineer_Resume_3.pdf",
"Machine_Learning_Engineer_Resume_2.pdf",
"Machine_Learning_Engineer_Resume_5.pdf",
"Machine_Learning_Engineer_Resume_4.pdf",
]

metadata = [
    dict(id=0, source=filenames[0]),
    dict(id=1, source=filenames[1]),
    dict(id=2, source=filenames[2]),
    dict(id=3, source=filenames[3]),
    dict(id=4, source=filenames[4]),
    dict(id=5, source=filenames[5]),
    dict(id=6, source=filenames[6]),
    dict(id=7, source=filenames[7]),
    dict(id=8, source=filenames[8]),
    dict(id=9, source=filenames[9]),
    dict(id=10, source=filenames[10]),
    dict(id=11, source=filenames[11]),
    dict(id=12, source=filenames[12]),
    dict(id=13, source=filenames[13]),
    dict(id=14, source=filenames[14]),
    # dict(year=2020, source=filenames[2]),
    # dict(year=2019, source=filenames[3])
    ]

data_root = "./data/"
# %%
# Process the documents
documents = []
for idx, file in enumerate(filenames):
    loader = PyPDFLoader(data_root + file)
    document = loader.load()
    for document_fragment in document:
        document_fragment.metadata = metadata[idx]
    documents += document
# %%
documents
# %%
# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Calculate and print interesting statistics
avg_doc_length = lambda docs: sum(len(doc.page_content) for doc in docs) // len(docs)
print(f'Average length among {len(documents)} documents loaded is {avg_doc_length(documents)} characters.')
print(f'After the split we have {len(docs)} documents as opposed to the original {len(documents)}.')
print(f'Average length among {len(docs)} documents (after split) is {avg_doc_length(docs)} characters.')

# Generate sample embedding for a document chunk
sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
print("Sample embedding of a document chunk: ", sample_embedding)
print("Size of the embedding: ", sample_embedding.shape)
# %%
# Create vector store and index wrapper
vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)
# %%
# Perform a query and print the result
query = "find top candidates for data scientist position?"
answer = wrapper_store_faiss.query(question=query, llm=llm)
print_ww(answer)

# %%
# Perform a more customizable query using RetrievalQA
prompt_template_str = """
Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Try to match candidate with following job description criteria
Professional Summary
Experienced Machine Learning Engineer with a strong background in designing, building, and deploying scalable
machine learning models. Adept at data engineering and deploying AI solutions into production environments. Looking
to leverage deep learning expertise to tackle new challenges in a dynamic team setting.
Education
M.S. in Computer Science, Specialization in Machine Learning
Relevant Coursework: Deep Learning, Advanced Machine Learning, Distributed Systems

{context}

Question: {question}

Assistant:"""
# PROMPT = PromptTemplate
# Create an instance of PromptTemplate
prompt_instance = PromptTemplate.from_template(prompt_template_str)
# %%
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_instance},
    callbacks=[token_counter]
)
# %%
# questions
# 1. "recommend good data scientist candidates?"
# 2. "How was Amazon impacted by COVID-19?"
# 3. "recommend good data machine learning engineer?"

# %%
query = "How was Amazon impacted by COVID-19?"
result = qa({"query": query})
print_ww(result['result'])

print(f"\n{result['source_documents']}")
# %%
