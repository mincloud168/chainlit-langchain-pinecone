from sentence_transformers import SentenceTransformer    
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

IMAGE_PINECONE_API_KEY = os.getenv("IMAGE_PINECONE_API_KEY")
IMAGE_PINECONE_ENV = os.getenv("IMAGE_PINECONE_ENV")

# initialize connection (get API key at app.pinecone.io)
def initPinecone(index_name:str, pinecone_api_key:str, pinecone_env:str, dimension_len:int = 1536) -> pinecone.Index:
    pinecone.init(
        api_key=pinecone_api_key,
        #environment="asia-southeast1-gcp"  # find next to API key  #us-central1-gcp #asia-southeast1-gcp
        environment=pinecone_env
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

def search(index:pinecone.Index ,query:str):
    model = SentenceTransformer('clip-ViT-B-32')

    # Encode text query
    query_string = query
    text_emb = model.encode(query_string)

    
    text_emb_list = text_emb.tolist()
    print(text_emb_list)
    results = index.query(vector=text_emb_list,top_k=1,include_metadata=True)
    print(results)

def main():
    index_name = "osha-images"
     
    index = initPinecone(index_name=index_name,pinecone_api_key=IMAGE_PINECONE_API_KEY,pinecone_env=IMAGE_PINECONE_ENV,dimension_len=512)
    search(index=index,query="building")  

if __name__ == "__main__":
    main()