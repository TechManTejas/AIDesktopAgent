"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List
import subprocess


async def run_shell_command(command: str) -> str:

    """ Execute a shell command in local host and return its output """
    try:
        # Execute the command and capture the output
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred while executing the command: {e}"
    

# async def search(
#     query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
# ) #-> Optional[list[dict[str, Any]]]:
#     """Search for general web results.

#     This function performs a search using the Tavily search engine, which is designed
#     to provide comprehensive, accurate, and trusted results. It's particularly useful
#     for answering questions about current events.
#     """
#     configuration = Configuration.from_runnable_config(config)
#     wrapped = TavilySearchResults(max_results=configuration.max_search_results)
#     result = await wrapped.ainvoke({"query": query})
#     #return cast(list[dict[str, Any]], result)



TOOLS: List[Callable[..., Any]] = [ run_shell_command ]
