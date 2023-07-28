from langchain import HuggingFaceHub, OpenAI
from langchain import PromptTemplate, LLMChain

from lab import query_pinecone,construtPrompt
import os

from dotenv import load_dotenv
import chainlit as cl

# Load environment variables from .env file
load_dotenv()

#HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

prompt_template = """
You are a helpful AI assistant and provide the answer for the question asked politely.

{question}
"""

index_name = "mtnet-faq-index"

@cl.langchain_factory(use_async=False)
def main():
    llm = OpenAI(temperature=0)
    """
    repo_id = "tiiuae/falcon-7b-instruct"
    llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                        repo_id=repo_id, 
                        model_kwargs={"temperature":0.7, "max_new_tokens":500})
    """

    

    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
    return chain


# synchronous 
@cl.langchain_run
async def run(agent, input_str):
    #
    contexts = query_pinecone(query=input_str,index_name=index_name,text_key="content")
    prompt_contexts = construtPrompt(query=input_str,contexts=contexts)
    
    res = await cl.make_async(agent)(prompt_contexts, callbacks=[cl.ChainlitCallbackHandler()])
    await cl.Message(content=res["text"]).send()

"""
@cl.langchain_run
async def run(agent, input_str):
    res = await agent.acall(input_str, callbacks=[cl.AsyncChainlitCallbackHandler()])
    await cl.Message(content=res).send()
"""