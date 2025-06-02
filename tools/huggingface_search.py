"""
Hugging Face Hub Search Tool
===========================

This module implements search functionality for the Hugging Face Hub
to discover relevant models and datasets for the research project.
"""

import requests
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class ModelInfo:
    """Data class for model information."""
    model_id: str
    author: str
    task: str
    description: str
    downloads: int
    likes: int
    tags: List[str]
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = None

@dataclass
class DatasetInfo:
    """Data class for dataset information."""
    dataset_id: str
    author: str
    description: str
    downloads: int
    likes: int
    tags: List[str]
    task_categories: List[str]

class HuggingFaceSearchTool:
    """
    Tool for searching Hugging Face Hub for models and datasets.
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the Hugging Face search tool.
        
        Args:
            token (str): Hugging Face API token (optional)
        """
        self.token = token
        self.base_url = "https://huggingface.co/api"
        self.headers = {"User-Agent": "research-agent/1.0"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    def search_models(self, query: str, task: Optional[str] = None, 
                     library: Optional[str] = None, limit: int = 10) -> List[ModelInfo]:
        """
        Search for models on Hugging Face Hub.
        
        Args:
            query (str): Search query
            task (str): Specific task filter (e.g., 'text-classification')
            library (str): Library filter (e.g., 'pytorch', 'transformers')
            limit (int): Maximum number of results
            
        Returns:
            List[ModelInfo]: List of model information
        """
        try:
            params = {
                "search": query,
                "limit": limit,
                "sort": "downloads",
                "direction": -1
            }
            
            if task:
                params["pipeline_tag"] = task
            if library:
                params["library"] = library
            
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                models_data = response.json()
                models = []
                
                for model_data in models_data:
                    model = ModelInfo(
                        model_id=model_data.get("id", ""),
                        author=model_data.get("author", ""),
                        task=model_data.get("pipeline_tag", "unknown"),
                        description=model_data.get("description", ""),
                        downloads=model_data.get("downloads", 0),
                        likes=model_data.get("likes", 0),
                        tags=model_data.get("tags", []),
                        pipeline_tag=model_data.get("pipeline_tag"),
                        library_name=model_data.get("library_name")
                    )
                    models.append(model)
                
                return models
            else:
                print(f"HF API error: {response.status_code}")
                return self._fallback_model_search(query, limit)
                
        except Exception as e:
            print(f"Model search error: {str(e)}")
            return self._fallback_model_search(query, limit)
    
    def search_datasets(self, query: str, task: Optional[str] = None, 
                       limit: int = 10) -> List[DatasetInfo]:
        """
        Search for datasets on Hugging Face Hub.
        
        Args:
            query (str): Search query
            task (str): Task category filter
            limit (int): Maximum number of results
            
        Returns:
            List[DatasetInfo]: List of dataset information
        """
        try:
            params = {
                "search": query,
                "limit": limit,
                "sort": "downloads",
                "direction": -1
            }
            
            if task:
                params["task_categories"] = task
            
            response = requests.get(
                f"{self.base_url}/datasets",
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                datasets_data = response.json()
                datasets = []
                
                for dataset_data in datasets_data:
                    dataset = DatasetInfo(
                        dataset_id=dataset_data.get("id", ""),
                        author=dataset_data.get("author", ""),
                        description=dataset_data.get("description", ""),
                        downloads=dataset_data.get("downloads", 0),
                        likes=dataset_data.get("likes", 0),
                        tags=dataset_data.get("tags", []),
                        task_categories=dataset_data.get("task_categories", [])
                    )
                    datasets.append(dataset)
                
                return datasets
            else:
                print(f"HF API error: {response.status_code}")
                return self._fallback_dataset_search(query, limit)
                
        except Exception as e:
            print(f"Dataset search error: {str(e)}")
            return self._fallback_dataset_search(query, limit)
    
    def _fallback_model_search(self, query: str, limit: int) -> List[ModelInfo]:
        """
        Fallback model search when API is not available.
        
        Args:
            query (str): Search query
            limit (int): Number of results
            
        Returns:
            List[ModelInfo]: Mock model results
        """
        print(f"Using fallback model search for query: {query}")
        
        # Common model suggestions based on query keywords
        mock_models = []
        
        if any(word in query.lower() for word in ['classification', 'text', 'nlp']):
            mock_models.extend([
                ModelInfo(
                    model_id="bert-base-uncased",
                    author="google",
                    task="text-classification",
                    description="BERT base model (uncased) for text classification tasks",
                    downloads=1000000,
                    likes=500,
                    tags=["pytorch", "tf", "text-classification"],
                    pipeline_tag="text-classification",
                    library_name="transformers"
                ),
                ModelInfo(
                    model_id="distilbert-base-uncased",
                    author="huggingface",
                    task="text-classification", 
                    description="Distilled version of BERT for faster inference",
                    downloads=800000,
                    likes=400,
                    tags=["pytorch", "tf", "distillation"],
                    pipeline_tag="text-classification",
                    library_name="transformers"
                )
            ])
        
        if any(word in query.lower() for word in ['vision', 'image', 'cnn']):
            mock_models.extend([
                ModelInfo(
                    model_id="google/vit-base-patch16-224",
                    author="google",
                    task="image-classification",
                    description="Vision Transformer (ViT) model for image classification",
                    downloads=500000,
                    likes=300,
                    tags=["pytorch", "vision", "transformer"],
                    pipeline_tag="image-classification",
                    library_name="transformers"
                ),
                ModelInfo(
                    model_id="microsoft/resnet-50",
                    author="microsoft",
                    task="image-classification",
                    description="ResNet-50 model pre-trained on ImageNet",
                    downloads=600000,
                    likes=250,
                    tags=["pytorch", "vision", "resnet"],
                    pipeline_tag="image-classification",
                    library_name="transformers"
                )
            ])
        
        return mock_models[:limit]
    
    def _fallback_dataset_search(self, query: str, limit: int) -> List[DatasetInfo]:
        """
        Fallback dataset search when API is not available.
        
        Args:
            query (str): Search query
            limit (int): Number of results
            
        Returns:
            List[DatasetInfo]: Mock dataset results
        """
        print(f"Using fallback dataset search for query: {query}")
        
        mock_datasets = [
            DatasetInfo(
                dataset_id=f"sample-{query.replace(' ', '-')}-dataset",
                author="research-community",
                description=f"Sample dataset for {query} research and experimentation",
                downloads=10000,
                likes=50,
                tags=["research", "benchmark"],
                task_categories=["classification"]
            )
        ]
        
        return mock_datasets[:limit]
    
    def get_model_details(self, model_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            Dict: Detailed model information
        """
        try:
            response = requests.get(
                f"{self.base_url}/models/{model_id}",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            print(f"Error getting model details: {str(e)}")
            return None
    
    def format_models_for_agent(self, models: List[ModelInfo]) -> str:
        """
        Format model search results for the agent.
        
        Args:
            models (List[ModelInfo]): Model search results
            
        Returns:
            str: Formatted results string
        """
        if not models:
            return "No models found."
        
        formatted = "Hugging Face Models Found:\n\n"
        for i, model in enumerate(models, 1):
            formatted += f"{i}. **{model.model_id}**\n"
            formatted += f"   Author: {model.author}\n"
            formatted += f"   Task: {model.task}\n"
            formatted += f"   Downloads: {model.downloads:,}\n"
            formatted += f"   Likes: {model.likes}\n"
            if model.description:
                formatted += f"   Description: {model.description[:200]}...\n"
            formatted += f"   Tags: {', '.join(model.tags[:5])}\n"
            formatted += f"   URL: https://huggingface.co/{model.model_id}\n\n"
        
        return formatted
    
    def format_datasets_for_agent(self, datasets: List[DatasetInfo]) -> str:
        """
        Format dataset search results for the agent.
        
        Args:
            datasets (List[DatasetInfo]): Dataset search results
            
        Returns:
            str: Formatted results string
        """
        if not datasets:
            return "No datasets found."
        
        formatted = "Hugging Face Datasets Found:\n\n"
        for i, dataset in enumerate(datasets, 1):
            formatted += f"{i}. **{dataset.dataset_id}**\n"
            formatted += f"   Author: {dataset.author}\n"
            formatted += f"   Downloads: {dataset.downloads:,}\n"
            formatted += f"   Likes: {dataset.likes}\n"
            if dataset.description:
                formatted += f"   Description: {dataset.description[:200]}...\n"
            formatted += f"   Task Categories: {', '.join(dataset.task_categories)}\n"
            formatted += f"   URL: https://huggingface.co/datasets/{dataset.dataset_id}\n\n"
        
        return formatted

# Example usage and testing
if __name__ == "__main__":
    # Test the HF search tool
    hf_tool = HuggingFaceSearchTool()
    
    # Test model search
    models = hf_tool.search_models("text classification", limit=3)
    print("Model Search Results:")
    print(hf_tool.format_models_for_agent(models))
    
    # Test dataset search
    datasets = hf_tool.search_datasets("classification", limit=2)
    print("\nDataset Search Results:")
    print(hf_tool.format_datasets_for_agent(datasets)) 