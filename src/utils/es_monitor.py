import logging
import json
from typing import Dict, Any, Optional
import time
from elasticsearch import Elasticsearch
import config

# Configure logger
logger = logging.getLogger(__name__)

class ElasticsearchMonitor:
    """Utility for monitoring Elasticsearch health and performance"""
    
    def __init__(self, es_client: Elasticsearch = None):
        """
        Initialize with an existing Elasticsearch client or create a new one.
        
        Args:
            es_client: An existing Elasticsearch client instance
        """
        self.es_client = es_client
        
        # Create a new client if none was provided
        if not self.es_client:
            connection_params = {}
            
            # Set hosts if provided
            if config.ELASTICSEARCH_HOSTS:
                connection_params["hosts"] = config.ELASTICSEARCH_HOSTS
                
            # Set cloud_id if provided
            if config.ELASTICSEARCH_CLOUD_ID:
                connection_params["cloud_id"] = config.ELASTICSEARCH_CLOUD_ID
                
            # Set authentication
            if config.ELASTICSEARCH_API_KEY:
                connection_params["api_key"] = config.ELASTICSEARCH_API_KEY
            elif config.ELASTICSEARCH_USERNAME and config.ELASTICSEARCH_PASSWORD:
                connection_params["basic_auth"] = (config.ELASTICSEARCH_USERNAME, config.ELASTICSEARCH_PASSWORD)
                
            try:
                self.es_client = Elasticsearch(**connection_params)
            except Exception as e:
                logger.error(f"Failed to create Elasticsearch client: {str(e)}")
                self.es_client = None
    
    def get_cluster_health(self) -> Optional[Dict[str, Any]]:
        """
        Get cluster health information.
        
        Returns:
            dict: Cluster health information
        """
        if not self.es_client:
            logger.error("Elasticsearch client not available")
            return None
            
        try:
            health = self.es_client.cluster.health()
            return health
        except Exception as e:
            logger.error(f"Failed to get cluster health: {str(e)}")
            return None
    
    def get_index_stats(self, index_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Get statistics for an index or all indices.
        
        Args:
            index_name: The name of the index to get stats for, or None for all indices
            
        Returns:
            dict: Index statistics
        """
        if not self.es_client:
            logger.error("Elasticsearch client not available")
            return None
            
        try:
            index_name = index_name or config.INDEX_NAME
            stats = self.es_client.indices.stats(index=index_name)
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return None
    
    def get_index_mapping(self, index_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Get mapping for an index.
        
        Args:
            index_name: The name of the index to get mapping for
            
        Returns:
            dict: Index mapping
        """
        if not self.es_client:
            logger.error("Elasticsearch client not available")
            return None
            
        try:
            index_name = index_name or config.INDEX_NAME
            mapping = self.es_client.indices.get_mapping(index=index_name)
            return mapping
        except Exception as e:
            logger.error(f"Failed to get index mapping: {str(e)}")
            return None
    
    def get_document_count(self, index_name: str = None) -> Optional[int]:
        """
        Get the number of documents in an index.
        
        Args:
            index_name: The name of the index to count documents in
            
        Returns:
            int: Number of documents
        """
        if not self.es_client:
            logger.error("Elasticsearch client not available")
            return None
            
        try:
            index_name = index_name or config.INDEX_NAME
            count = self.es_client.count(index=index_name)
            return count["count"]
        except Exception as e:
            logger.error(f"Failed to get document count: {str(e)}")
            return None
    
    def benchmark_search(self, query: str, iterations: int = 5) -> Optional[Dict[str, Any]]:
        """
        Benchmark search performance.
        
        Args:
            query: The search query to benchmark
            iterations: Number of iterations to perform
            
        Returns:
            dict: Benchmark results including average, min, and max times
        """
        if not self.es_client:
            logger.error("Elasticsearch client not available")
            return None
            
        try:
            index_name = config.INDEX_NAME
            
            # Prepare search body
            search_body = {
                "query": {
                    "match": {
                        "patterns.primary_pattern": {
                            "query": query,
                            "fuzziness": "AUTO"
                        }
                    }
                },
                "size": 20
            }
            
            # Track times
            times = []
            
            # Run benchmark
            for i in range(iterations):
                start_time = time.time()
                self.es_client.search(index=index_name, body=search_body)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate stats
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            return {
                "query": query,
                "iterations": iterations,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "times": times
            }
        except Exception as e:
            logger.error(f"Failed to benchmark search: {str(e)}")
            return None
    
    def get_storage_usage(self, index_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Get storage usage information for an index.
        
        Args:
            index_name: The name of the index to get storage info for
            
        Returns:
            dict: Storage usage information
        """
        if not self.es_client:
            logger.error("Elasticsearch client not available")
            return None
            
        try:
            index_name = index_name or config.INDEX_NAME
            stats = self.es_client.indices.stats(index=index_name, metric="store")
            
            # Format storage info
            if index_name in stats["indices"]:
                store_info = stats["indices"][index_name]["total"]["store"]
                return {
                    "size_in_bytes": store_info["size_in_bytes"],
                    "size_in_mb": store_info["size_in_bytes"] / (1024 * 1024),
                    "size_in_gb": store_info["size_in_bytes"] / (1024 * 1024 * 1024)
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get storage usage: {str(e)}")
            return None
    
    def print_health_report(self) -> None:
        """Print a comprehensive health report"""
        if not self.es_client:
            logger.error("Elasticsearch client not available")
            return
            
        try:
            # Get cluster health
            health = self.get_cluster_health()
            if health:
                print("\n=== Elasticsearch Cluster Health ===")
                print(f"Status: {health.get('status', 'unknown')}")
                print(f"Number of nodes: {health.get('number_of_nodes', 0)}")
                print(f"Active shards: {health.get('active_shards', 0)}")
                print(f"Relocating shards: {health.get('relocating_shards', 0)}")
                print(f"Initializing shards: {health.get('initializing_shards', 0)}")
                print(f"Unassigned shards: {health.get('unassigned_shards', 0)}")
            
            # Get index stats
            index_name = config.INDEX_NAME
            doc_count = self.get_document_count(index_name)
            storage = self.get_storage_usage(index_name)
            
            print(f"\n=== '{index_name}' Index Stats ===")
            print(f"Documents: {doc_count or 'unknown'}")
            if storage:
                print(f"Size: {storage.get('size_in_mb', 0):.2f} MB")
            
            # Benchmark a simple search
            benchmark = self.benchmark_search("test")
            if benchmark:
                print("\n=== Search Performance ===")
                print(f"Average search time: {benchmark.get('avg_time', 0) * 1000:.2f} ms")
                print(f"Min search time: {benchmark.get('min_time', 0) * 1000:.2f} ms")
                print(f"Max search time: {benchmark.get('max_time', 0) * 1000:.2f} ms")
            
            print("\n=== Recommendations ===")
            
            # Make recommendations based on health
            if health and health.get('status') == 'yellow':
                print("- The cluster status is yellow. Consider adding more nodes or reallocating shards.")
            
            if health and health.get('unassigned_shards', 0) > 0:
                print("- There are unassigned shards. Check Elasticsearch logs for allocation issues.")
            
            if doc_count is not None and doc_count > 10000:
                print("- Consider optimizing your index for large datasets:")
                print("  - Increase the number of shards for better distribution")
                print("  - Use more specific mappings to reduce index size")
                print("  - Enable caching for frequent queries")
            
            print("")
        except Exception as e:
            logger.error(f"Failed to print health report: {str(e)}")
    
if __name__ == "__main__":
    # If script is run directly, print health report
    monitor = ElasticsearchMonitor()
    monitor.print_health_report() 