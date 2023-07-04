import numpy as np
import openai
import pandas as pd
import pickle
import uuid
import pinecone
from tqdm.auto import tqdm
import datetime
from time import sleep
import traceback
from dotenv import load_dotenv
import os

load_dotenv()

IMAGE_PINECONE_API_KEY = os.getenv("IMAGE_PINECONE_API_KEY")
IMAGE_PINECONE_ENV = os.getenv("IMAGE_PINECONE_ENV")

# initialize connection (get API key at app.pinecone.io)
def initPinecone(index_name:str, pinecone_api_key:str, env:str, dimension_len:int = 1536) -> pinecone.Index:
    pinecone.init(
        api_key=pinecone_api_key,
        #environment="asia-southeast1-gcp"  # find next to API key  #us-central1-gcp #asia-southeast1-gcp
        environment=env
    )
        #environment="us-central1-gcp"  # find next to API key
    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
    # if does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=dimension_len,
            metric='cosine',
            metadata_config={
                'indexed': ['title', 'heading']
            }
        )

    # connect to index
    return pinecone.Index(index_name)




