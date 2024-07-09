import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.agents import Tool, AgentType, initialize_agent

from langchain_core.output_parsers import StrOutputParser
print(dotenv.load_dotenv())
chat_model = ChatGoogleGenerativeAI(model="gemini-pro")
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain import hub
gemini_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

from langchain_intro.tools import get_current_wait_time
from langchain_core.messages import HumanMessage, SystemMessage
from langchain import hub, LLMMathChain
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
REVIEWS_CHROMA_PATH = "chroma_data/"
review_template_str = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

{context}
"""
review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=gemini_embedding
)
reviews_retriever  = reviews_vector_db.as_retriever(k=10)

review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)
tools = [
    Tool(
        name="Reviews",
        func=review_chain.invoke,
        description="""Useful when you need to answer questions
        about patient reviews or experiences at the hospital.
        Not useful for answering questions about specific visit
        details such as payer, billing, treatment, diagnosis,
        chief complaint, hospital, or physician information.
        Pass the entire question as input to the tool. For instance,
        if the question is "What do patients think about the triage system?",
        the input should be "What do patients think about the triage system?"
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_time,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. This tool returns wait times in
        minutes. Do not pass the word "hospital" as input,
        only the hospital name itself. For instance, if the question is
        "What is the wait time at hospital A?", the input should be "A".
        """,
    ),
]
agent_prompt = hub.pull("mikechan/gemini")
agent_prompt.template

# Wanted to create agent function like openai for gemini but it not seems to work 
#openai: article agent with langchain https://realpython.com/build-llm-rag-chatbot-with-langchain/
#gemini: https://github.com/MikeChan-HK/Gemini-agent-example/blob/main/Gemini_agents.ipynb
# hospital_agent = create_openai_functions_agent(
#     llm=agent_chat_model,
#     prompt=hospital_agent_prompt,
#     tools=tools,
# )
agent = ( # not seems to work
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | agent_prompt,
    | llm_with_stop=chat_model,
    | ReActSingleInputOutputParser()
)
