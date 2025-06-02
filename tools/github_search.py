"""
GitHub Search Tool
=================

This module implements search functionality for GitHub repositories
to find relevant code implementations and examples.
"""

import requests
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import base64

@dataclass
class RepositoryInfo:
    """Data class for repository information."""
    name: str
    full_name: str
    description: str
    url: str
    stars: int
    forks: int
    language: str
    topics: List[str]
    updated_at: str
    size: int

@dataclass
class CodeFile:
    """Data class for code file information."""
    name: str
    path: str
    content: str
    size: int
    download_url: str

class GitHubSearchTool:
    """
    Tool for searching GitHub repositories and code files.
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the GitHub search tool.
        
        Args:
            token (str): GitHub API token (optional, but recommended for higher rate limits)
        """
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "research-agent/1.0"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"
    
    def search_repositories(self, query: str, language: Optional[str] = None, 
                          sort: str = "stars", limit: int = 10) -> List[RepositoryInfo]:
        """
        Search for repositories on GitHub.
        
        Args:
            query (str): Search query
            language (str): Programming language filter
            sort (str): Sort by 'stars', 'forks', 'updated'
            limit (int): Maximum number of results
            
        Returns:
            List[RepositoryInfo]: List of repository information
        """
        try:
            search_query = query
            if language:
                search_query += f" language:{language}"
            
            params = {
                "q": search_query,
                "sort": sort,
                "order": "desc",
                "per_page": min(limit, 100)  # GitHub API limit
            }
            
            response = requests.get(
                f"{self.base_url}/search/repositories",
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                repositories = []
                
                for repo_data in data.get("items", []):
                    repo = RepositoryInfo(
                        name=repo_data.get("name", ""),
                        full_name=repo_data.get("full_name", ""),
                        description=repo_data.get("description", ""),
                        url=repo_data.get("html_url", ""),
                        stars=repo_data.get("stargazers_count", 0),
                        forks=repo_data.get("forks_count", 0),
                        language=repo_data.get("language", ""),
                        topics=repo_data.get("topics", []),
                        updated_at=repo_data.get("updated_at", ""),
                        size=repo_data.get("size", 0)
                    )
                    repositories.append(repo)
                
                return repositories
            else:
                print(f"GitHub API error: {response.status_code}")
                return self._fallback_repository_search(query, limit)
                
        except Exception as e:
            print(f"Repository search error: {str(e)}")
            return self._fallback_repository_search(query, limit)
    
    def search_code(self, query: str, language: Optional[str] = None, 
                   filename: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Search for code files on GitHub.
        
        Args:
            query (str): Search query
            language (str): Programming language filter
            filename (str): Filename filter
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: List of code search results
        """
        try:
            search_query = query
            if language:
                search_query += f" language:{language}"
            if filename:
                search_query += f" filename:{filename}"
            
            params = {
                "q": search_query,
                "per_page": min(limit, 100)
            }
            
            response = requests.get(
                f"{self.base_url}/search/code",
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("items", [])
            else:
                print(f"GitHub code search error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Code search error: {str(e)}")
            return []
    
    def get_file_content(self, repo_full_name: str, file_path: str) -> Optional[CodeFile]:
        """
        Get the content of a specific file from a repository.
        
        Args:
            repo_full_name (str): Repository full name (owner/repo)
            file_path (str): Path to the file in the repository
            
        Returns:
            CodeFile: File content and metadata
        """
        try:
            response = requests.get(
                f"{self.base_url}/repos/{repo_full_name}/contents/{file_path}",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Decode base64 content
                content = ""
                if data.get("encoding") == "base64":
                    try:
                        content = base64.b64decode(data.get("content", "")).decode('utf-8')
                    except:
                        content = "Binary file or encoding error"
                
                file_info = CodeFile(
                    name=data.get("name", ""),
                    path=data.get("path", ""),
                    content=content,
                    size=data.get("size", 0),
                    download_url=data.get("download_url", "")
                )
                
                return file_info
            else:
                return None
                
        except Exception as e:
            print(f"Error getting file content: {str(e)}")
            return None
    
    def get_repository_structure(self, repo_full_name: str, path: str = "") -> List[Dict]:
        """
        Get the directory structure of a repository.
        
        Args:
            repo_full_name (str): Repository full name (owner/repo)
            path (str): Path within the repository
            
        Returns:
            List[Dict]: Directory contents
        """
        try:
            url = f"{self.base_url}/repos/{repo_full_name}/contents"
            if path:
                url += f"/{path}"
            
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return []
                
        except Exception as e:
            print(f"Error getting repository structure: {str(e)}")
            return []
    
    def _fallback_repository_search(self, query: str, limit: int) -> List[RepositoryInfo]:
        """
        Fallback repository search when API is not available.
        
        Args:
            query (str): Search query
            limit (int): Number of results
            
        Returns:
            List[RepositoryInfo]: Mock repository results
        """
        print(f"Using fallback repository search for query: {query}")
        
        # Generate mock repositories based on common patterns
        mock_repos = []
        
        if any(word in query.lower() for word in ['machine learning', 'ml', 'classification']):
            mock_repos.extend([
                RepositoryInfo(
                    name="awesome-machine-learning",
                    full_name="josephmisiti/awesome-machine-learning",
                    description="A curated list of awesome Machine Learning frameworks, libraries and software.",
                    url="https://github.com/josephmisiti/awesome-machine-learning",
                    stars=50000,
                    forks=12000,
                    language="Python",
                    topics=["machine-learning", "awesome-list"],
                    updated_at="2024-01-01T00:00:00Z",
                    size=1000
                ),
                RepositoryInfo(
                    name="scikit-learn",
                    full_name="scikit-learn/scikit-learn",
                    description="scikit-learn: machine learning in Python",
                    url="https://github.com/scikit-learn/scikit-learn",
                    stars=45000,
                    forks=20000,
                    language="Python",
                    topics=["machine-learning", "python", "scikit-learn"],
                    updated_at="2024-01-01T00:00:00Z",
                    size=50000
                )
            ])
        
        if any(word in query.lower() for word in ['deep learning', 'neural', 'pytorch', 'tensorflow']):
            mock_repos.extend([
                RepositoryInfo(
                    name="pytorch",
                    full_name="pytorch/pytorch",
                    description="Tensors and Dynamic neural networks in Python with strong GPU acceleration",
                    url="https://github.com/pytorch/pytorch",
                    stars=60000,
                    forks=15000,
                    language="Python",
                    topics=["deep-learning", "pytorch", "neural-networks"],
                    updated_at="2024-01-01T00:00:00Z",
                    size=100000
                )
            ])
        
        return mock_repos[:limit]
    
    def format_repositories_for_agent(self, repositories: List[RepositoryInfo]) -> str:
        """
        Format repository search results for the agent.
        
        Args:
            repositories (List[RepositoryInfo]): Repository search results
            
        Returns:
            str: Formatted results string
        """
        if not repositories:
            return "No repositories found."
        
        formatted = "GitHub Repositories Found:\n\n"
        for i, repo in enumerate(repositories, 1):
            formatted += f"{i}. **{repo.name}** ({repo.full_name})\n"
            formatted += f"   Description: {repo.description}\n"
            formatted += f"   Language: {repo.language}\n"
            formatted += f"   Stars: {repo.stars:,} | Forks: {repo.forks:,}\n"
            if repo.topics:
                formatted += f"   Topics: {', '.join(repo.topics[:5])}\n"
            formatted += f"   URL: {repo.url}\n"
            formatted += f"   Last Updated: {repo.updated_at}\n\n"
        
        return formatted
    
    def format_code_results_for_agent(self, code_results: List[Dict]) -> str:
        """
        Format code search results for the agent.
        
        Args:
            code_results (List[Dict]): Code search results
            
        Returns:
            str: Formatted results string
        """
        if not code_results:
            return "No code files found."
        
        formatted = "Code Files Found:\n\n"
        for i, result in enumerate(code_results, 1):
            formatted += f"{i}. **{result.get('name', 'Unknown')}**\n"
            formatted += f"   Repository: {result.get('repository', {}).get('full_name', 'Unknown')}\n"
            formatted += f"   Path: {result.get('path', 'Unknown')}\n"
            formatted += f"   URL: {result.get('html_url', 'Unknown')}\n\n"
        
        return formatted

# Example usage and testing
if __name__ == "__main__":
    # Test the GitHub search tool
    github_tool = GitHubSearchTool()
    
    # Test repository search
    repos = github_tool.search_repositories("machine learning python", language="Python", limit=3)
    print("Repository Search Results:")
    print(github_tool.format_repositories_for_agent(repos))
    
    # Test code search
    code_results = github_tool.search_code("neural network training", language="Python", limit=2)
    print("\nCode Search Results:")
    print(github_tool.format_code_results_for_agent(code_results)) 