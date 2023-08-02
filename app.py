from langchain import HuggingFaceHub, OpenAI
from langchain import PromptTemplate, LLMChain

from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents.structured_chat.prompt import SUFFIX

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

@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0, streaming=True)
    """
    repo_id = "tiiuae/falcon-7b-instruct"
    llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                        repo_id=repo_id, 
                        model_kwargs={"temperature":0.7, "max_new_tokens":500})
    """
    tools = []
    memory = ConversationBufferMemory(memory_key="chat_history")
    _SUFFIX = "Chat history:\n{chat_history}\n\n" + SUFFIX

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        agent_kwargs={
            "suffix": _SUFFIX,
            "input_variables": ["input", "agent_scratchpad", "chat_history"],
        },
    )
    cl.user_session.set("agent", agent)


# synchronous 
@cl.on_message
async def main(message):
    #
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    contexts = query_pinecone(query=message,index_name=index_name,text_key="content")
    prompt_contexts = construtPrompt(query=message,contexts=contexts)
    

    res = await cl.make_async(agent.run)(
        input=prompt_contexts, callbacks=[cl.LangchainCallbackHandler()]
    )
    elements = []
    actions = []
    print(res)
    await cl.Message(content=res, elements=elements, actions=actions).send()
    #await cl.Message(content=res["text"]).send()

"""
@cl.langchain_run
async def run(agent, input_str):
    res = await agent.acall(input_str, callbacks=[cl.AsyncChainlitCallbackHandler()])
    await cl.Message(content=res).send()
"""