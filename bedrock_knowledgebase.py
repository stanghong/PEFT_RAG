# # %%
# %pip install -U opensearch-py==2.3.1
# %pip install -U boto3==1.33.2
# %pip install -U retrying==1.3.4
# %%
# restart kernel
from IPython.core.display import HTML
HTML("<script>Jupyter.notebook.kernel.restart()</script>")
# %%
import warnings
warnings.filterwarnings('ignore')
# %%
import json
import os
import boto3
import pprint
from utility import create_bedrock_execution_role, create_oss_policy_attach_bedrock_execution_role, create_policies_in_oss
import random
from retrying import retry
suffix =644 #random.randrange(200, 900)

boto3_session = boto3.session.Session()
region_name = boto3_session.region_name
bedrock_agent_client = boto3_session.client('bedrock-agent', region_name=region_name)
service = 'aoss'
bucket_name = "test-bucket-aoss" # replace it with your bucket name.
pp = pprint.PrettyPrinter(indent=2)
# %%
# Create VectorStore
import boto3
import time
vector_store_name = f'bedrock-sample-rag-{suffix}'
index_name = f"bedrock-sample-rag-index-{suffix}"
aoss_client = boto3_session.client('opensearchserverless')
bedrock_kb_execution_role = create_bedrock_execution_role(bucket_name=bucket_name)
bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']
# %%
# create security, network and data access policies within OSS
encryption_policy, network_policy, access_policy = create_policies_in_oss(vector_store_name=vector_store_name,
                       aoss_client=aoss_client,
                       bedrock_kb_execution_role_arn=bedrock_kb_execution_role_arn
                    )
collection = aoss_client.create_collection(name=vector_store_name,type='VECTORSEARCH')
# %%pp.pprint(collection)
time.sleep(10)

# %%
collection_id = collection['createCollectionDetail']['id']
host = collection_id + '.' + region_name + '.aoss.amazonaws.com'
print(host)
# %%
# create oss policy and attach it to Bedrock execution role
create_oss_policy_attach_bedrock_execution_role(collection_id=collection_id,
                                                bedrock_kb_execution_role=bedrock_kb_execution_role)
# %%
# Step 2 - Create vector index
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
credentials = boto3.Session().get_credentials()
awsauth = auth = AWSV4SignerAuth(credentials, region_name, service)

index_name = f"bedrock-sample-index-{suffix}"
body_json = {
   "settings": {
      "index.knn": "true"
   },
   "mappings": {
      "properties": {
         "vector": {
            "type": "knn_vector",
            "dimension": 1536
         },
         "text": {
            "type": "text"
         },
         "text-metadata": {
            "type": "text"         }
      }
   }
}
# Build the OpenSearch client
oss_client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=300
)
# # It can take up to a minute for data access rules to be enforced
time.sleep(60)

# %%
# Create index
response = oss_client.indices.create(index=index_name, body=json.dumps(body_json))
print('\nCreating index:')
print(response)

# %%
# Download and prepare dataset
# !mkdir -p ./data

# from urllib.request import urlretrieve
import os

def list_pdf_files(directory):
    """
    Lists all the PDF files in the given directory.

    :param directory: Path to the directory
    :return: List of PDF file names
    """
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    return pdf_files

# Replace 'your_directory_path' with the path to the directory you want to scan.
data_root = "./data/"
filenames = list_pdf_files(data_root)

print("PDF files in the directory:", filenames)

# %%

for idx, url in enumerate(filenames):
    file_path = data_root + filenames[idx]
file_path 
    # urlretrieve(url, file_path)
# %%
# create bucket
import boto3
from botocore.exceptions import ClientError

def create_bucket(bucket_name, region=None):
    """
    Create an S3 bucket in a specified region.

    :param bucket_name: Bucket to create
    :param region: String region to create bucket in, e.g., 'us-west-2'
    :return: True if bucket created, else False
    """
    try:
        if region is None or region == 'us-east-1':
            s3_client = boto3.client('s3')
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
    except ClientError as e:
        print(f"Error: {e}")
        return False
    return True
# %%
# Usage

region = 'us-east-1'  # replace with your desired region
created = create_bucket(bucket_name, region)

if created:
    print(f"Bucket {bucket_name} created successfully.")
else:
    print(f"Could not create bucket {bucket_name}.")
# %%
# Upload data to s3
s3_client = boto3.client("s3")
def uploadDirectory(path,bucket_name):
        for root,dirs,files in os.walk(path):
            for file in files:
                s3_client.upload_file(os.path.join(root,file),bucket_name,file)

uploadDirectory(data_root, bucket_name)

# %%
# Create KB
opensearchServerlessConfiguration = {
            "collectionArn": collection["createCollectionDetail"]['arn'],
            "vectorIndexName": index_name,
            "fieldMapping": {
                "vectorField": "vector",
                "textField": "text",
                "metadataField": "text-metadata"
            }
        }

chunkingStrategyConfiguration = {
    "chunkingStrategy": "FIXED_SIZE",
    "fixedSizeChunkingConfiguration": {
        "maxTokens": 512,
        "overlapPercentage": 20
    }
}

s3Configuration = {
    "bucketArn": f"arn:aws:s3:::{bucket_name}",
    # "inclusionPrefixes":["*.*"] # you can use this if you want to create a KB using data within s3 prefixes.
}

embeddingModelArn = f"arn:aws:bedrock:{region_name}::foundation-model/amazon.titan-embed-text-v1"

name = f"bedrock-sample-knowledge-base-{suffix}"
description = "Amazon shareholder letter knowledge base."
roleArn = bedrock_kb_execution_role_arn
# %%
# Create a KnowledgeBase
from retrying import retry

@retry(wait_random_min=1000, wait_random_max=2000,stop_max_attempt_number=7)
def create_knowledge_base_func():
    create_kb_response = bedrock_agent_client.create_knowledge_base(
        name = name,
        description = description,
        roleArn = roleArn,
        knowledgeBaseConfiguration = {
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": embeddingModelArn
            }
        },
        storageConfiguration = {
            "type": "OPENSEARCH_SERVERLESS",
            "opensearchServerlessConfiguration":opensearchServerlessConfiguration
        }
    )
    return create_kb_response["knowledgeBase"]
# %%
try:
    kb = create_knowledge_base_func()
except Exception as err:
    print(f"{err=}, {type(err)=}")
# %%
# Get KnowledgeBase 
get_kb_response = bedrock_agent_client.get_knowledge_base(knowledgeBaseId = kb['knowledgeBaseId'])
# %%
# Create a DataSource in KnowledgeBase 
create_ds_response = bedrock_agent_client.create_data_source(
    name = name,
    description = description,
    knowledgeBaseId = kb['knowledgeBaseId'],
    dataSourceConfiguration = {
        "type": "S3",
        "s3Configuration":s3Configuration
    },
    vectorIngestionConfiguration = {
        "chunkingConfiguration": chunkingStrategyConfiguration
    }
)
ds = create_ds_response["dataSource"]
pp.pprint(ds)
# %%
# Get DataSource 
bedrock_agent_client.get_data_source(knowledgeBaseId = kb['knowledgeBaseId'], dataSourceId = ds["dataSourceId"])
# %%
# Get DataSource 
bedrock_agent_client.get_data_source(knowledgeBaseId = kb['knowledgeBaseId'], dataSourceId = ds["dataSourceId"])
# %%
# Start an ingestion job
start_job_response = bedrock_agent_client.start_ingestion_job(knowledgeBaseId = kb['knowledgeBaseId'], dataSourceId = ds["dataSourceId"])
# %%
job = start_job_response["ingestionJob"]
pp.pprint(job)
# %%
# Get job 
while(job['status']!='COMPLETE' ):
  get_job_response = bedrock_agent_client.get_ingestion_job(
      knowledgeBaseId = kb['knowledgeBaseId'],
        dataSourceId = ds["dataSourceId"],
        ingestionJobId = job["ingestionJobId"]
  )
  job = get_job_response["ingestionJob"]
pp.pprint(job)
time.sleep(40)
# %%
kb_id = kb["knowledgeBaseId"]
pp.pprint(kb_id)
# %%
%store kb_id
# %%
# try out KB using RetrieveAndGenerate API
bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=region_name)
model_id = "anthropic.claude-instant-v1" # try with both claude instant as well as claude-v2. for claude v2 - "anthropic.claude-v2"
model_arn = f'arn:aws:bedrock:us-east-1::foundation-model/{model_id}'
# %%
# query = "What is mission cloud team vacation time?"
query = "What is the order from amazon?"
response = bedrock_agent_runtime_client.retrieve_and_generate(
    input={
        'text': query
    },
    retrieveAndGenerateConfiguration={
        'type': 'KNOWLEDGE_BASE',
        'knowledgeBaseConfiguration': {
            'knowledgeBaseId': kb_id,
            'modelArn': model_arn
        }
    },
)

generated_text = response['output']['text']
pp.pprint(generated_text)
# %%
## print out the source attribution/citations from the original documents to see if the response generated belongs to the context.
citations = response["citations"]
contexts = []
for citation in citations:
    retrievedReferences = citation["retrievedReferences"]
    for reference in retrievedReferences:
        contexts.append(reference["content"]["text"])

pp.pprint(contexts)
# %%
# retreive api for fetching only the relevant context.
relevant_documents = bedrock_agent_runtime_client.retrieve(
    retrievalQuery= {
        'text': query
    },
    knowledgeBaseId=kb_id,
    retrievalConfiguration= {
        'vectorSearchConfiguration': {
            'numberOfResults': 3 # will fetch top 3 documents which matches closely with the query.
        }
    }
)
# %%

pp.pprint(relevant_documents["retrievalResults"])
# %%



# %%
# Delete KnowledgeBase
bedrock_agent_client.delete_data_source(dataSourceId = ds["dataSourceId"], knowledgeBaseId=kb['knowledgeBaseId'])
bedrock_agent_client.delete_knowledge_base(knowledgeBaseId=kb['knowledgeBaseId'])
oss_client.indices.delete(index=index_name)
aoss_client.delete_collection(id=collection_id)
aoss_client.delete_access_policy(type="data", name=access_policy['accessPolicyDetail']['name'])
aoss_client.delete_security_policy(type="network", name=network_policy['securityPolicyDetail']['name'])
aoss_client.delete_security_policy(type="encryption", name=encryption_policy['securityPolicyDetail']['name'])
# delete role and policies
from utility import delete_iam_role_and_policies
delete_iam_role_and_policies()
