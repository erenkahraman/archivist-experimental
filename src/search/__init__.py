"""
Search package that provides functionality for document indexing and retrieval.
"""
from .elasticsearch_client import ElasticsearchClient
from .search_engine import search_engine

__all__ = ['ElasticsearchClient', 'search_engine'] 