from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage, ToolCallRequestEvent, ToolCallExecutionEvent
import asyncio
import arxiv
from pprint import pprint
import creds

'''
Code from Mohammad Hossein Amini
'''

def searchArxiv(query: str, max_results: int = 5, sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance) -> list:
    """
    Search for papers on arxiv.org using the arxiv API.
    
    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.
        sort_by (str): The sorting criteria for the results, e.g., "relevance" or "newest".
        
    Returns:
        list: A list of dictionaries containing paper information
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
    )
    results = client.results(search)
    papers = []

    for res in results:
        papers.append({
            "title": res.title,
            "summary": res.summary,
            "authors": [author.name for author in res.authors],
            "id": res.entry_id,
            "url": res.pdf_url,
        })
    return papers

def teamConfig():
    model = OpenAIChatCompletionClient(
        model="o3-mini",
        api_key=creds.api_key,
    )

    arxiv_agent = AssistantAgent(
        name="arxiv_agent",
        system_message=(
            "You are a helpful assistant that gets a description from user and "
            "searches for the most relevant or newest papers on arxiv.org using python arxiv API. "
            "Always search for five times more papers than the user requested. Use the tool "
            "provided to conduct the search. Craft the query in arxiv API format."
        ),
        model_client=model,
        tools=[searchArxiv],
        reflect_on_tool_use=True,
    )

    researcher = AssistantAgent(
        name="researcher",
        system_message=(
            "You are a researcher agent. Based on the papers provided by the arxiv agent, "
            "You need to generate a markdown report that summarizes the papers. Initially, "
            "you need to generate an introduction that describes the research area and then, "
            "for each paper, you need to state the title, the authors, the link to the paper, "
            "the problem they are solving, and how they are solving it. "
        ),
        model_client=model,
    ) 

    team = RoundRobinGroupChat(
        participants=[arxiv_agent, researcher],
        max_turns=2,
    )

    return team
    
async def orchestrate(team, task):
    async for msg in team.run_stream(task=task):
        print("--" * 20)
        if isinstance(msg, TextMessage):
            print(message:=f'{msg.source}: {msg.content}')
            yield message
        elif isinstance(msg, ToolCallRequestEvent):
            print(message:=str(msg))
            yield message
        elif isinstance(msg, ToolCallExecutionEvent):
            print(message:=str(msg))
            yield message

async def main(task):
    team = teamConfig()
    async for message in orchestrate(team, task):
        pass

if __name__ == '__main__':
    task = "Find five best papers on GAN for image generation."
    asyncio.run(main(task))
