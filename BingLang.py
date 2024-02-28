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
from datetime import datetime  
import pytz  


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
        #Some pages don't return expected result so we use a try except method to avoid getting an errors. 
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

def get_time_in_timezone(timezone):  
    try:  
        # Create a timezone object using pytz  
        tz = pytz.timezone(timezone)  
          
        # Get the current time in that timezone  
        current_time = datetime.now(tz)  
          
        # Format the time to a string if necessary  
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')  
          
        return time_str  
    except pytz.exceptions.UnknownTimeZoneError:  
        return "Unknown timezone. Please provide a valid timezone."  

def get_weather(city_name):   

    base_url = "http://api.weatherapi.com/v1/current.json"  
    api_key = os.getenv("Weather_API")
    complete_url = f"{base_url}?key={api_key}&q={city_name}"  
  
    response = requests.get(complete_url)  
  
    # Check if the request was successful  
    if response.status_code == 200:  
        weather_data = response.json()  
        return weather_data  
    else:  
        return "Failed to get weather data."  

# Configuring the prompt, memeory, and LLM for the chatbot 
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are an AI assistance who can access the internet through bing_search tool and to time information via check_time tool. 
            The bing_search tool will return the webpage content that contains information that you can use to answer user's question. 
            Whenever asked about time and date use the check_time tool and for weather related questions use check_weather tool and just provide a short answer. 
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

tool2 = Tool(
    name="check_time",
    description="Used to return country's time",
    func=get_time_in_timezone
)

tool3 = Tool(
    name="check_weather", 
    description="Used to find weather information about a city", 
    func=get_weather
)

# Creating the langchain agent: 
agent = create_openai_tools_agent(llm, [tool], prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[tool, tool2, tool3],
    verbose=True, #Set to true to view the thought process in the AOAI model.
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
