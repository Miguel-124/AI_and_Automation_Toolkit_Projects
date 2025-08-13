from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(temperature=0)

@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmeric calculations with numbers"""
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a + b}"
    
@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user"""
    print("Tool has been called.")
    return f"Hello {name}, I hope you are well today"

@tool("chat", return_direct=True, description="Answer any user query by forwarding it to the ChatOpenAI model.")
def chat_tool(query: str) -> str:
    return model.invoke([HumanMessage(content=query)]).content

def main():
    
    tools = [calculator, say_hello, chat_tool]
    agent_executor = create_react_agent(model, tools)
    
    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ["quit", "exit"]:
            break
        
        print("\nAssistant: ", end="")
        for chunk in agent_executor.stream({"messages":[HumanMessage(content=user_input)]}):
            if "tools" in chunk and "messages" in chunk["tools"]:
                for msg in chunk["tools"]["messages"]:
                    print(msg.content, end="")

        print()
        
if __name__ == "__main__":
    main()
                