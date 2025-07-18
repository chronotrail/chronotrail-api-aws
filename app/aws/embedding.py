"""
Embedding and vector storage service for ChronoTrail API.

This module provides functionality for generating text embeddings using Amazon Bedrock
and storing/searching them in AWS OpenSearch, with proper user isolation and error handling.
"""
import json
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union
from uuid import UUID
from datetime import datetime

from fastapi import HTTPException, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger
from app.aws.clients import (
    get_bedrock_embeddings_client,
    get_opensearch_client,
    handle_aws_error,
    handle_aws_error_async,
    with_retry,
)
from app.aws.exceptions import BedrockError, OpenSearchError
from app.models.schemas import ContentType, VectorSearchResult

# Configure logger
logger = get_logger(__name__)

# Constants for embedding and vector search
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"  # Amazon Titan Embeddings model
EMBEDDING_DIMENSION = 1536  # Dimension of the embedding vectors
OPENSEARCH_INDEX_NAME = "chronotrail-content"  # Name of the OpenSearch index
OPENSEARCH_MAPPING = {
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},  # For user isolation
            "content_type": {"type": "keyword"},  # Type of content
            "content_text": {"type": "text"},  # Original text content
            "content_vector": {
                "type": "knn_vector",
                "dimension": EMBEDDING_DIMENSION,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib"
                }
            },
            "timestamp": {"type": "date"},  # When the content was created
            "source_id": {"type": "keyword"},  # Reference to relational DB record
            "location": {
                "properties": {
                    "longitude": {"type": "float"},
                    "latitude": {"type": "float"},
                    "address": {"type": "text"},
                    "names": {"type": "keyword"}  # Array of location names/tags
                }
            },
            "metadata": {
                "properties": {
                    "file_type": {"type": "keyword"},
                    "original_filename": {"type": "keyword"}
                }
            }
        }
    },
    "settings": {
        "index": {
            "number_of_shards": 3,
            "number_of_replicas": 1,
            "refresh_interval": "1s"
        }
    }
}


class EmbeddingService:
    """
    Service for generating text embeddings and managing vector storage.
    
    This service provides methods for generating embeddings using Amazon Bedrock,
    storing them in AWS OpenSearch, and performing vector similarity searches
    with proper user isolation.
    """
    
    def __init__(self, index_name: Optional[str] = None):
        """
        Initialize the embedding service.
        
        Args:
            index_name: Optional OpenSearch index name (defaults to OPENSEARCH_INDEX_NAME)
        """
        self.index_name = index_name or OPENSEARCH_INDEX_NAME
        logger.info(f"Initialized EmbeddingService with index: {self.index_name}")
    
    @handle_aws_error(service_name='bedrock-runtime')
    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for text using Amazon Bedrock.
        
        Args:
            text: Text to generate embeddings for
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            BedrockError: If embedding generation fails
        """
        try:
            # Get Bedrock client
            bedrock_client = get_bedrock_embeddings_client()
            
            # Prepare request body
            request_body = json.dumps({
                "inputText": text
            })
            
            # Call Bedrock API
            response = bedrock_client.invoke_model(
                modelId=EMBEDDING_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=request_body
            )
            
            # Parse response
            response_body = json.loads(response.get('body').read())
            embedding = response_body.get('embedding')
            
            if not embedding:
                raise BedrockError(
                    message="No embedding found in Bedrock response",
                    operation="generate_embeddings"
                )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise BedrockError(
                message=f"Failed to generate embeddings: {str(e)}",
                operation="generate_embeddings"
            )
    
    @handle_aws_error(service_name='bedrock-runtime')
    async def generate_embeddings_async(self, text: str) -> List[float]:
        """
        Generate embeddings for text using Amazon Bedrock (async wrapper).
        
        This is a wrapper around the synchronous method since Bedrock doesn't have
        native async support in boto3 yet.
        
        Args:
            text: Text to generate embeddings for
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            BedrockError: If embedding generation fails
        """
        # For now, just call the synchronous method
        # In the future, this could be implemented with a thread pool
        # We need to make sure we don't try to await a non-coroutine
        return self.generate_embeddings(text)
    
    @with_retry(max_attempts=3, base_delay=1.0)
    def ensure_index_exists(self) -> bool:
        """
        Ensure the OpenSearch index exists with proper mapping.
        
        Returns:
            bool: True if index exists or was created successfully
            
        Raises:
            OpenSearchError: If index creation fails
        """
        try:
            # Get OpenSearch client
            opensearch_client = get_opensearch_client()
            
            # Check if index exists
            if opensearch_client.indices.exists(index=self.index_name):
                logger.debug(f"OpenSearch index {self.index_name} already exists")
                return True
            
            # Create index with mapping
            opensearch_client.indices.create(
                index=self.index_name,
                body=OPENSEARCH_MAPPING
            )
            
            logger.info(f"Created OpenSearch index {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {str(e)}")
            raise OpenSearchError(
                message=f"Failed to ensure index exists: {str(e)}",
                operation="ensure_index_exists"
            )
    
    @handle_aws_error(service_name='opensearch')
    def index_document(
        self,
        user_id: Union[str, UUID],
        content_type: Union[str, ContentType],
        content_text: str,
        content_vector: List[float],
        timestamp: Optional[datetime] = None,
        source_id: Optional[Union[str, UUID]] = None,
        location: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Index a document in OpenSearch.
        
        Args:
            user_id: User ID for isolation
            content_type: Type of content
            content_text: Original text content
            content_vector: Embedding vector
            timestamp: When the content was created (defaults to now)
            source_id: Reference to relational DB record
            location: Optional location data
            metadata: Optional additional metadata
            
        Returns:
            str: Document ID
            
        Raises:
            OpenSearchError: If indexing fails
        """
        try:
            # Ensure index exists
            self.ensure_index_exists()
            
            # Convert UUID to string if needed
            if isinstance(user_id, UUID):
                user_id = str(user_id)
            
            if isinstance(source_id, UUID):
                source_id = str(source_id)
            
            # Convert ContentType enum to string if needed
            if isinstance(content_type, ContentType):
                content_type = content_type.value
            
            # Generate document ID if not provided
            doc_id = str(uuid.uuid4())
            
            # Prepare document
            document = {
                "user_id": user_id,
                "content_type": content_type,
                "content_text": content_text,
                "content_vector": content_vector,
                "timestamp": timestamp.isoformat() if timestamp else datetime.utcnow().isoformat(),
                "source_id": source_id,
            }
            
            # Add location if provided
            if location:
                document["location"] = location
            
            # Add metadata if provided
            if metadata:
                document["metadata"] = metadata
            
            # Get OpenSearch client
            opensearch_client = get_opensearch_client()
            
            # Index document
            response = opensearch_client.index(
                index=self.index_name,
                id=doc_id,
                body=document,
                refresh=True  # Make document immediately searchable
            )
            
            logger.debug(f"Indexed document {doc_id} in {self.index_name}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to index document: {str(e)}")
            raise OpenSearchError(
                message=f"Failed to index document: {str(e)}",
                operation="index_document"
            )
    
    @handle_aws_error(service_name='opensearch')
    async def index_document_async(
        self,
        user_id: Union[str, UUID],
        content_type: Union[str, ContentType],
        content_text: str,
        content_vector: List[float],
        timestamp: Optional[datetime] = None,
        source_id: Optional[Union[str, UUID]] = None,
        location: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Index a document in OpenSearch (async wrapper).
        
        This is a wrapper around the synchronous method since OpenSearch Python client
        doesn't have native async support.
        
        Args:
            user_id: User ID for isolation
            content_type: Type of content
            content_text: Original text content
            content_vector: Embedding vector
            timestamp: When the content was created (defaults to now)
            source_id: Reference to relational DB record
            location: Optional location data
            metadata: Optional additional metadata
            
        Returns:
            str: Document ID
            
        Raises:
            OpenSearchError: If indexing fails
        """
        # For now, just call the synchronous method
        # In the future, this could be implemented with a thread pool
        return self.index_document(
            user_id=user_id,
            content_type=content_type,
            content_text=content_text,
            content_vector=content_vector,
            timestamp=timestamp,
            source_id=source_id,
            location=location,
            metadata=metadata
        )
    
    @handle_aws_error(service_name='opensearch')
    def search_by_vector(
        self,
        user_id: Union[str, UUID],
        vector: List[float],
        k: int = 10,
        content_types: Optional[List[Union[str, ContentType]]] = None,
        min_score: float = 0.7,
        date_range: Optional[Dict[str, datetime]] = None,
        location_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents by vector similarity.
        
        Args:
            user_id: User ID for isolation
            vector: Query vector
            k: Number of results to return
            content_types: Optional filter by content types
            min_score: Minimum similarity score (0-1)
            date_range: Optional date range filter
            location_filter: Optional location filter
            
        Returns:
            List[Dict[str, Any]]: Search results
            
        Raises:
            OpenSearchError: If search fails
        """
        try:
            # Convert UUID to string if needed
            if isinstance(user_id, UUID):
                user_id = str(user_id)
            
            # Convert ContentType enums to strings if needed
            if content_types:
                content_types = [
                    ct.value if isinstance(ct, ContentType) else ct
                    for ct in content_types
                ]
            
            # Build query
            query = {
                "size": k,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "term": {
                                    "user_id": user_id
                                }
                            }
                        ],
                        "should": [
                            {
                                "knn": {
                                    "content_vector": {
                                        "vector": vector,
                                        "k": k
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                }
            }
            
            # Add content type filter if provided
            if content_types:
                query["query"]["bool"]["must"].append({
                    "terms": {
                        "content_type": content_types
                    }
                })
            
            # Add date range filter if provided
            if date_range:
                date_filter = {}
                if "start_date" in date_range:
                    date_filter["gte"] = date_range["start_date"].isoformat()
                if "end_date" in date_range:
                    date_filter["lte"] = date_range["end_date"].isoformat()
                
                if date_filter:
                    query["query"]["bool"]["must"].append({
                        "range": {
                            "timestamp": date_filter
                        }
                    })
            
            # Add location filter if provided
            if location_filter:
                if "coordinates" in location_filter:
                    lon = location_filter["coordinates"].get("longitude")
                    lat = location_filter["coordinates"].get("latitude")
                    distance = location_filter.get("distance", "10km")
                    
                    if lon is not None and lat is not None:
                        query["query"]["bool"]["must"].append({
                            "geo_distance": {
                                "distance": distance,
                                "location": {
                                    "lon": lon,
                                    "lat": lat
                                }
                            }
                        })
                
                if "names" in location_filter:
                    query["query"]["bool"]["must"].append({
                        "terms": {
                            "location.names": location_filter["names"]
                        }
                    })
            
            # Get OpenSearch client
            opensearch_client = get_opensearch_client()
            
            # Execute search
            response = opensearch_client.search(
                index=self.index_name,
                body=query
            )
            
            # Process results
            hits = response.get("hits", {}).get("hits", [])
            results = []
            
            for hit in hits:
                score = hit.get("_score", 0)
                if score < min_score:
                    continue
                
                source = hit.get("_source", {})
                results.append({
                    "id": hit.get("_id"),
                    "score": score,
                    "content_type": source.get("content_type"),
                    "content_text": source.get("content_text"),
                    "timestamp": source.get("timestamp"),
                    "source_id": source.get("source_id"),
                    "location": source.get("location"),
                    "metadata": source.get("metadata")
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search by vector: {str(e)}")
            raise OpenSearchError(
                message=f"Failed to search by vector: {str(e)}",
                operation="search_by_vector"
            )
    
    @handle_aws_error(service_name='opensearch')
    async def search_by_vector_async(
        self,
        user_id: Union[str, UUID],
        vector: List[float],
        k: int = 10,
        content_types: Optional[List[Union[str, ContentType]]] = None,
        min_score: float = 0.7,
        date_range: Optional[Dict[str, datetime]] = None,
        location_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents by vector similarity (async wrapper).
        
        This is a wrapper around the synchronous method since OpenSearch Python client
        doesn't have native async support.
        
        Args:
            user_id: User ID for isolation
            vector: Query vector
            k: Number of results to return
            content_types: Optional filter by content types
            min_score: Minimum similarity score (0-1)
            date_range: Optional date range filter
            location_filter: Optional location filter
            
        Returns:
            List[Dict[str, Any]]: Search results
            
        Raises:
            OpenSearchError: If search fails
        """
        # For now, just call the synchronous method
        # In the future, this could be implemented with a thread pool
        return self.search_by_vector(
            user_id=user_id,
            vector=vector,
            k=k,
            content_types=content_types,
            min_score=min_score,
            date_range=date_range,
            location_filter=location_filter
        )
    
    @handle_aws_error(service_name='opensearch')
    def search_by_text(
        self,
        user_id: Union[str, UUID],
        text: str,
        k: int = 10,
        content_types: Optional[List[Union[str, ContentType]]] = None,
        date_range: Optional[Dict[str, datetime]] = None,
        location_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents by text similarity.
        
        This method first generates embeddings for the query text, then searches
        for similar documents using vector similarity.
        
        Args:
            user_id: User ID for isolation
            text: Query text
            k: Number of results to return
            content_types: Optional filter by content types
            date_range: Optional date range filter
            location_filter: Optional location filter
            
        Returns:
            List[Dict[str, Any]]: Search results
            
        Raises:
            OpenSearchError: If search fails
            BedrockError: If embedding generation fails
        """
        try:
            # Generate embeddings for query text
            vector = self.generate_embeddings(text)
            
            # Search by vector
            return self.search_by_vector(
                user_id=user_id,
                vector=vector,
                k=k,
                content_types=content_types,
                date_range=date_range,
                location_filter=location_filter
            )
            
        except (OpenSearchError, BedrockError) as e:
            logger.error(f"Failed to search by text: {str(e)}")
            raise
    
    @handle_aws_error(service_name='opensearch')
    async def search_by_text_async(
        self,
        user_id: Union[str, UUID],
        text: str,
        k: int = 10,
        content_types: Optional[List[Union[str, ContentType]]] = None,
        date_range: Optional[Dict[str, datetime]] = None,
        location_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents by text similarity (async wrapper).
        
        Args:
            user_id: User ID for isolation
            text: Query text
            k: Number of results to return
            content_types: Optional filter by content types
            date_range: Optional date range filter
            location_filter: Optional location filter
            
        Returns:
            List[Dict[str, Any]]: Search results
            
        Raises:
            OpenSearchError: If search fails
            BedrockError: If embedding generation fails
        """
        try:
            # Generate embeddings for query text - not awaiting since it's not a coroutine
            vector = self.generate_embeddings_async(text)
            
            # Search by vector - not awaiting since it's not a coroutine
            return self.search_by_vector_async(
                user_id=user_id,
                vector=vector,
                k=k,
                content_types=content_types,
                date_range=date_range,
                location_filter=location_filter
            )
            
        except (OpenSearchError, BedrockError) as e:
            logger.error(f"Failed to search by text: {str(e)}")
            raise
    
    @handle_aws_error(service_name='opensearch')
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from OpenSearch.
        
        Args:
            doc_id: Document ID
            
        Returns:
            bool: True if document was deleted successfully
            
        Raises:
            OpenSearchError: If deletion fails
        """
        try:
            # Get OpenSearch client
            opensearch_client = get_opensearch_client()
            
            # Delete document
            response = opensearch_client.delete(
                index=self.index_name,
                id=doc_id,
                refresh=True
            )
            
            result = response.get("result") == "deleted"
            logger.debug(f"Deleted document {doc_id} from {self.index_name}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            raise OpenSearchError(
                message=f"Failed to delete document: {str(e)}",
                operation="delete_document"
            )
    
    @handle_aws_error(service_name='opensearch')
    def delete_user_documents(self, user_id: Union[str, UUID]) -> int:
        """
        Delete all documents for a user from OpenSearch.
        
        Args:
            user_id: User ID
            
        Returns:
            int: Number of documents deleted
            
        Raises:
            OpenSearchError: If deletion fails
        """
        try:
            # Convert UUID to string if needed
            if isinstance(user_id, UUID):
                user_id = str(user_id)
            
            # Get OpenSearch client
            opensearch_client = get_opensearch_client()
            
            # Delete by query
            response = opensearch_client.delete_by_query(
                index=self.index_name,
                body={
                    "query": {
                        "term": {
                            "user_id": user_id
                        }
                    }
                },
                refresh=True
            )
            
            deleted = response.get("deleted", 0)
            logger.info(f"Deleted {deleted} documents for user {user_id} from {self.index_name}")
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete user documents: {str(e)}")
            raise OpenSearchError(
                message=f"Failed to delete user documents: {str(e)}",
                operation="delete_user_documents"
            )
    
    async def process_and_store_text(
        self,
        user_id: Union[str, UUID],
        content_type: Union[str, ContentType],
        text: str,
        source_id: Optional[Union[str, UUID]] = None,
        timestamp: Optional[datetime] = None,
        location: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process text and store it in the vector database.
        
        This method generates embeddings for the text and stores it in OpenSearch.
        
        Args:
            user_id: User ID for isolation
            content_type: Type of content
            text: Text to process
            source_id: Reference to relational DB record
            timestamp: When the content was created (defaults to now)
            location: Optional location data
            metadata: Optional additional metadata
            
        Returns:
            str: Document ID
            
        Raises:
            BedrockError: If embedding generation fails
            OpenSearchError: If indexing fails
        """
        try:
            # Generate embeddings - not awaiting since it's not a coroutine
            vector = self.generate_embeddings_async(text)
            
            # Index document - not awaiting since it's not a coroutine
            doc_id = self.index_document_async(
                user_id=user_id,
                content_type=content_type,
                content_text=text,
                content_vector=vector,
                timestamp=timestamp,
                source_id=source_id,
                location=location,
                metadata=metadata
            )
            
            return doc_id
            
        except (BedrockError, OpenSearchError) as e:
            logger.error(f"Failed to process and store text: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing text: {str(e)}")
            raise OpenSearchError(
                message=f"Unexpected error processing text: {str(e)}",
                operation="process_and_store_text"
            )
    
    def convert_search_results_to_model(
        self,
        results: List[Dict[str, Any]]
    ) -> List[VectorSearchResult]:
        """
        Convert raw search results to VectorSearchResult models.
        
        Args:
            results: Raw search results
            
        Returns:
            List[VectorSearchResult]: Formatted search results
        """
        vector_results = []
        
        for result in results:
            # Parse timestamp
            timestamp = datetime.fromisoformat(result["timestamp"]) if result.get("timestamp") else datetime.utcnow()
            
            # Parse source_id
            source_id = None
            if result.get("source_id"):
                try:
                    source_id = UUID(result["source_id"])
                except (ValueError, TypeError):
                    source_id = None
            
            # Create VectorSearchResult
            vector_result = VectorSearchResult(
                content_id=result["id"],
                content_type=result["content_type"],
                content_text=result["content_text"],
                timestamp=timestamp,
                score=result["score"],
                source_id=source_id,
                location=result.get("location")
            )
            
            vector_results.append(vector_result)
        
        return vector_results


# Create a singleton instance
embedding_service = EmbeddingService()