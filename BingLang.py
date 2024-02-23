import os
from dotenv import load_dotenv
from langchain_community.utilities import BingSearchAPIWrapper
from langchain.tools import Tool
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from bs4 import BeautifulSoup
import requests


load_dotenv()

search = BingSearchAPIWrapper()

# The bingsearch snippet doesn't always provide enough information.
# I am getting the whole webpage content to feed it to the GPT to answer user's questions. 
def WebContent(query):
    """
    This tool is used to return the WebPage contents and can be used to answer user's questions. 
    """
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36'}
    results = search.results(query, 3) #Number of webpages to check and return content. 
    links = []
    contents = []
    for result in results: 
        try:
            webpage = requests.get(result['link'], headers)
            soup = BeautifulSoup(webpage.content, 'html.parser')
            text = soup.find('body').get_text().strip()
            cleaned_text = ' '.join(text.split('\n'))
            cleaned_text = ' '.join(text.split())
            contents.append(cleaned_text)
            links.append(result['link'])
        except:
            continue
    return contents, links

# Configuring the prompt, memeory, and LLM for the chatbot 
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are an AI assistance who can access the internet through bing_search tool. 
            The bing_search tool will return the webpage content that contains information that you can use to answer user's question. 
            Whenever asked about time, weather, and date use the bing_search tool and just provide a short answer. 
            For other questions provide a max of one paragraph unless instructed otherwise.
            If you get blocked or access deny, try another query for the bing_search tool to get access to a different website.
            Talk a bit to the user while grabbing the result. 
            Always provide the link in the following format. "\nUsed this link: {link} to answer your question. 
            """
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        MessagesPlaceholder(
            variable_name='agent_scratchpad'
        ),  # where tools are loaded for intermediate steps.
        HumanMessagePromptTemplate.from_template(
            "{input}"
        ),  # Where the human input will injected
    ]
)
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k= 8) #Chat memory window that keeps k messages. 
llm = AzureChatOpenAI(azure_deployment=os.getenv("Chat_deployment"), streaming=True)
tool = Tool(
    name="bing_search",
    description="Search Bing for recent results.",
    func=WebContent
)

# Creating the langchain agent: 
agent = create_openai_tools_agent(llm, [tool], prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[tool],
    verbose=True,
    memory=memory,
    max_iterations= 8 # Number of tries to retrieve data before exiting agent. 
)

def main():
    question = input("What do you like to ask?\n")
    while "exit" not in question.lower():  
        answer = agent_executor.invoke({"input": question})
        print(answer['output'])  
        question = input("\nDo you have other queries you would like to know about? if not type exit to end the chat.\n")  
    print(memory.load_memory_variables({})) #print the chat history. 

if __name__ == "__main__":
    main()