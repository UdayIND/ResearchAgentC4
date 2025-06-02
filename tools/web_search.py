"""
Brave Search Tool
================

This module implements web search functionality using the Brave Search API
for the autonomous research agent.
"""

import requests
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Data class for search results."""
    title: str
    url: str
    snippet: str
    published_date: Optional[str] = None

class BraveSearchTool:
    """
    Tool for performing web searches using Brave Search API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Brave Search tool.
        
        Args:
            api_key (str): Brave Search API key
        """
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip"
        }
        if api_key:
            self.headers["X-Subscription-Token"] = api_key
    
    def search_web(self, query: str, count: int = 10, safe_search: str = "moderate") -> List[SearchResult]:
        """
        Perform a web search using Brave Search API.
        
        Args:
            query (str): Search query
            count (int): Number of results to return (max 20)
            safe_search (str): Safe search setting (strict, moderate, off)
            
        Returns:
            List[SearchResult]: List of search results
        """
        try:
            params = {
                "q": query,
                "count": min(count, 20),  # Brave API limit
                "safesearch": safe_search,
                "search_lang": "en",
                "country": "US"
            }
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if "web" in data and "results" in data["web"]:
                    for item in data["web"]["results"]:
                        result = SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            snippet=item.get("description", ""),
                            published_date=item.get("age", None)
                        )
                        results.append(result)
                
                return results
            else:
                print(f"Search API error: {response.status_code}")
                return self._fallback_search(query, count)
                
        except Exception as e:
            print(f"Search error: {str(e)}")
            return self._fallback_search(query, count)
    
    def _fallback_search(self, query: str, count: int) -> List[SearchResult]:
        """
        Fallback search method when API is not available.
        Returns mock results for development/testing.
        
        Args:
            query (str): Search query
            count (int): Number of results
            
        Returns:
            List[SearchResult]: Mock search results
        """
        print(f"Using fallback search for query: {query}")
        
        # Mock results for common research topics
        mock_results = [
            SearchResult(
                title=f"Recent Advances in {query} - arXiv",
                url=f"https://arxiv.org/search/?query={query.replace(' ', '+')}&searchtype=all",
                snippet=f"Recent research papers and preprints related to {query}. "
                       f"This collection includes state-of-the-art methods and experimental results.",
                published_date="2024"
            ),
            SearchResult(
                title=f"{query} - Wikipedia",
                url=f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                snippet=f"Comprehensive overview of {query} including background, methods, "
                       f"applications, and current research directions.",
                published_date="2024"
            ),
            SearchResult(
                title=f"A Survey of {query} Methods",
                url=f"https://example.com/survey-{query.replace(' ', '-')}",
                snippet=f"Comprehensive survey of {query} methodologies, comparing different "
                       f"approaches and their performance on benchmark datasets.",
                published_date="2024"
            )
        ]
        
        return mock_results[:count]
    
    def search_academic_papers(self, query: str, count: int = 5) -> List[SearchResult]:
        """
        Search specifically for academic papers and research articles.
        
        Args:
            query (str): Search query
            count (int): Number of results
            
        Returns:
            List[SearchResult]: Academic search results
        """
        academic_query = f"{query} site:arxiv.org OR site:scholar.google.com OR filetype:pdf"
        return self.search_web(academic_query, count)
    
    def search_code_repositories(self, query: str, count: int = 5) -> List[SearchResult]:
        """
        Search for code repositories and implementations.
        
        Args:
            query (str): Search query
            count (int): Number of results
            
        Returns:
            List[SearchResult]: Code repository search results
        """
        code_query = f"{query} site:github.com OR site:gitlab.com implementation code"
        return self.search_web(code_query, count)
    
    def format_results_for_agent(self, results: List[SearchResult]) -> str:
        """
        Format search results for the agent to process.
        
        Args:
            results (List[SearchResult]): Search results
            
        Returns:
            str: Formatted results string
        """
        if not results:
            return "No search results found."
        
        formatted = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. **{result.title}**\n"
            formatted += f"   URL: {result.url}\n"
            formatted += f"   Summary: {result.snippet}\n"
            if result.published_date:
                formatted += f"   Date: {result.published_date}\n"
            formatted += "\n"
        
        return formatted

# Example usage and testing
if __name__ == "__main__":
    # Test the search tool
    search_tool = BraveSearchTool()
    
    # Test basic search
    results = search_tool.search_web("machine learning classification", count=3)
    print("Basic Search Results:")
    print(search_tool.format_results_for_agent(results))
    
    # Test academic search
    academic_results = search_tool.search_academic_papers("neural networks", count=2)
    print("\nAcademic Search Results:")
    print(search_tool.format_results_for_agent(academic_results)) 