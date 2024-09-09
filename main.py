import os
import sys
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_googledrive.tools.google_drive.tool import GoogleDriveSearchTool
from langchain_googledrive.utilities.google_drive import GoogleDriveAPIWrapper

# Function to save the API key to a .env file
def save_api_key_to_env(key_name, api_key):
    if os.path.exists(".env"):
        with open(".env", "a") as env_file:  # Append to the existing .env file
            env_file.write(f"{key_name}={api_key}\n")
    else:
        with open(".env", "w") as env_file:  # Create a new .env file if none exists
            env_file.write(f"{key_name}={api_key}\n")
    print(f"{key_name} saved to .env file.")

# Function to get and verify API keys
def get_api_key(key_name):
    # Load environment variables from the .env file
    load_dotenv()
    
    api_key = os.getenv(key_name)
    if not api_key:
        # Prompt the user to enter the API key if not found
        api_key = input(f"Enter your {key_name}: ")
        save_api_key_to_env(key_name, api_key)
        # Reload the environment to use the new API key
        load_dotenv()
        api_key = os.getenv(key_name)
    return api_key

# Get the environment variables for the API keys
anthropic_api_key = get_api_key("ANTHROPIC_API_KEY")
google_drive_api_key = get_api_key("GOOGLE_DRIVE_API_KEY")

# Initialize the Google Drive search tool
folder_id = "root"  # Use "root" or a specific folder ID

tool = GoogleDriveSearchTool(
    name="google_drive_search",
    api_wrapper=GoogleDriveAPIWrapper(
        folder_id=folder_id,
        num_results=2,
        template="gdrive-query-in-folder",
        api_key=google_drive_api_key  # Use the Google Drive API key
    )
)

# Create the agent
memory = MemorySaver()
model = ChatAnthropic(model_name="claude-3-sonnet-20240229", api_key=anthropic_api_key)

# Combine tools
tools = [tool]

agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Configuration for the agent
config = {"configurable": {"thread_id": "abc123"}}

# Use the agent in a loop
while True:
    prompt = input("Enter a prompt: ")

    # Use the agent to handle the user prompt
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=prompt)]}, config
    ):
        print(chunk)
        print("----")
