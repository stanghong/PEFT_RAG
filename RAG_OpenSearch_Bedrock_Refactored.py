# %%
# Import necessary libraries
import warnings
import json
import os
import sys
import boto3
import numpy as np
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.load.dump import dumps
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
import time

# %%
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww
# %%
# Suppress warnings
warnings.filterwarnings('ignore')

# Configure AWS environment variables
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
# %%
# Create a Bedrock client
boto3_bedrock = bedrock.get_bedrock_client(
    #assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

# Initialize the Bedrock and BedrockEmbeddings models
llm = Bedrock(
    model_id="anthropic.claude-v2",
    client=boto3_bedrock,
    model_kwargs={"max_tokens_to_sample": 200}
)
bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)
# %%
# List of PDF filenames and their metadata
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
    ]

data_root = "./data/"
# %%
# Process PDF documents

documents = []

for idx, file in enumerate(filenames):
    loader = PyPDFLoader(data_root + file)
    document = loader.load()
    for document_fragment in document:
        document_fragment.metadata = metadata[idx]
    documents += document
# %%
# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

docs = text_splitter.split_documents(documents)
# %%
# Calculate document statistics
avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents]) // len(documents)
avg_char_count_pre = avg_doc_length(documents)
avg_char_count_post = avg_doc_length(docs)
print(f"Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.")
print(f"After the split we have {len(docs)} documents more than the original {len(documents)}.")
print(f"Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters.")
# %%
# Embed a sample document chunk
try:
    sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
    modelId = bedrock_embeddings.model_id
    print("Embedding model Id :", modelId)
    print("Sample embedding of a document chunk: ", sample_embedding)
    print("Size of the embedding: ", sample_embedding.shape)
except ValueError as error:
    if "AccessDeniedException" in str(error):
        print(f"\x1b[41m{error}\
        \nTo troubleshoot this issue please refer to the following resources.\
         \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
         \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")      
        class StopExecution(ValueError):
            def _render_traceback_(self):
                pass
        raise StopExecution        
    else:
        raise error

# %% 
vector_store_name = 'clockwork-recruiting-rag'
index_name = "clockwork-recruiting-rag-index"
encryption_policy_name = "clockwork-recruiting-rag-sp"
network_policy_name = "clockwork-recruiting-rag-np"
access_policy_name = 'clockwork-recruiting-rag-ap'
identity = boto3.client('sts').get_caller_identity()['Arn']

# Your collection name
collection_name = 'clockwork-recruiting-rag'

# Initialize the OpenSearch Serverless client#
aoss_client = boto3.client('opensearchserverless')

# Create an AWS Identity
identity = boto3.client('sts').get_caller_identity()['Arn']

# %%
#####################################################################
#######################   create  policiess #########################
#####################################################################


# create policies
def create_policies(vector_store_name):
    security_policy = aoss_client.create_security_policy(
        name = encryption_policy_name,
        policy = json.dumps(
            {
                'Rules': [{'Resource': ['collection/' + vector_store_name],
                'ResourceType': 'collection'}],
                'AWSOwnedKey': True
            }),
        type = 'encryption'
    )

    network_policy = aoss_client.create_security_policy(
        name = network_policy_name,
        policy = json.dumps(
            [
                {'Rules': [{'Resource': ['collection/' + vector_store_name],
                'ResourceType': 'collection'}],
                'AllowFromPublic': True}
            ]),
        type = 'network'
    )

    access_policy = aoss_client.create_access_policy(
    name = access_policy_name,
    policy = json.dumps(
        [
            {
                'Rules': [
                    {
                        'Resource': ['collection/' + vector_store_name],
                        'Permission': [
                            'aoss:CreateCollectionItems',
                            'aoss:DeleteCollectionItems',
                            'aoss:UpdateCollectionItems',
                            'aoss:DescribeCollectionItems'],
                        'ResourceType': 'collection'
                    },
                    {
                        'Resource': ['index/' + vector_store_name + '/*'],
                        'Permission': [
                            'aoss:CreateIndex',
                            'aoss:DeleteIndex',
                            'aoss:UpdateIndex',
                            'aoss:DescribeIndex',
                            'aoss:ReadDocument',
                            'aoss:WriteDocument'],
                        'ResourceType': 'index'
                    }],
                'Principal': [identity],
                'Description': 'Easy data policy'}
        ]),
    type = 'data'
)
# %%
#####################################################################
####################        helper functions    #####################
#####################################################################
# Function to check if a collection exists
def collection_exists(collection_name):
    collections = aoss_client.list_collections()['collectionSummaries']
    return any(collection['name'] == collection_name for collection in collections)

# Function to create a collection
def create_collection(collection_name):
    collection = aoss_client.create_collection(
        name=collection_name,
        type='SEARCH',
        description='A collection for storing data'
    )
    return collection

def get_host(collection_name):
    if collection_exists(collection_name): # if not colelction exists
        response = aoss_client.list_collections(collectionFilters={'name': collection_name})
        host =  response['collectionSummaries'][0]['id'] + '.' + os.environ.get("AWS_DEFAULT_REGION", None) + '.aoss.amazonaws.com:443'
    else:
        collection=create_collection(collection_name)
        print(f"Collection '{collection_name}' created.")
    return host

# %%
# from langchain.schema.document import Document
# docs = [Document(page_content='xxx', metadata={'id': 0, 'source': 'xx'})]

# %%
#####################################################################
#######################   create  collection  #######################
#####################################################################
# %%
if not collection_exists(collection_name):
    # create_policies(collection_name)
    collection = aoss_client.create_collection(name=vector_store_name,type='VECTORSEARCH')

    response = aoss_client.list_collections(collectionFilters={'name': collection_name})
    host =  response['collectionSummaries'][0]['id'] + '.' + os.environ.get("AWS_DEFAULT_REGION", None) + '.aoss.amazonaws.com:443'
    print(f"Collection '{collection_name}' created.")

    service = 'aoss'
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, os.environ.get("AWS_DEFAULT_REGION", None), service)
    # ingest docs into opensearch and create index
    docsearch = OpenSearchVectorSearch.from_documents(
    docs,
    bedrock_embeddings,
    opensearch_url=host,
    http_auth=auth,
    timeout = 100,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    index_name=index_name,
    engine="faiss",
    )
# %%

#####################################################################
###############   read existing  collection   #######################
#####################################################################
# %%
if collection_exists(collection_name):
    response = aoss_client.list_collections(collectionFilters={'name': collection_name})
    host =  response['collectionSummaries'][0]['id'] + '.' + os.environ.get("AWS_DEFAULT_REGION", None) + '.aoss.amazonaws.com:443'

    service = 'aoss'
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, os.environ.get("AWS_DEFAULT_REGION", None), service)
    # read existing aoss collections
    docsearch = OpenSearchVectorSearch(
        embedding_function=bedrock_embeddings,
        opensearch_url=host,
        http_auth=auth,
        timeout = 100,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        index_name=index_name,
    )
    
# %%
# Perform question and answer retrieval
query = "find top 5 data scientists?"

results = docsearch.similarity_search(query, k=3)  # our search query  # return 3 most relevant docs
print(dumps(results, pretty=True))


##################################################################################################
# Prompt-based question answering with source information
# ...

# Clean up

# %%
# Conclusion and takeaways
# ...
aoss_client.delete_collection(id=collection_name['createCollectionDetail']['id'])
# %%

# aoss_client.delete_collection(id=collection_name['createCollectionDetail']['id'])
# aoss_client.delete_access_policy(name=access_policy_name, type='data')
# aoss_client.delete_security_policy(name=encryption_policy_name, type='encryption')
# aoss_client.delete_security_policy(name=network_policy_name, type='network')
# %%
