from langchain_intro.chatbot import agent_executer
from langchain_intro.chatbot import test_agent_execution
agent_executer.invoke(
    {"input": "What is the current wait time at hospital C?"}
)
agent_executer.invoke(
    {"input": "What have patients said about their comfort at the hospital?"}
)
# Test the "Waits" tool
test_agent_execution("Waits", "A")  # Should output 12