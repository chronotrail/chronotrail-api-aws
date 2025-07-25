"""
Query processing engine for ChronoTrail API.

This module provides functionality for processing natural language queries,
performing vector similarity search, structured data filtering, and result ranking
with proper user isolation and relevance scoring.
"""

import math
import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

from fastapi import HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.aws.embedding import EmbeddingService, embedding_service
from app.aws.llm import LLMService, llm_service
from app.core.config import settings
from app.core.logging import get_logger
from app.db.database import get_db
from app.models.database import User as DBUser
from app.models.schemas import (
    ContentType,
    FileType,
    MediaReference,
    QueryRequest,
    QueryResponse,
    User,
    VectorSearchResult,
)
from app.repositories.location_visits import location_visit_repository
from app.repositories.media_files import media_file_repository
from app.repositories.query_sessions import query_session_repository
from app.repositories.text_notes import text_note_repository
from app.services.usage import usage_service

# Configure logger
logger = get_logger(__name__)

# Constants for query processing
DEFAULT_VECTOR_SEARCH_K = 20
DEFAULT_MIN_SCORE = 0.7
DEFAULT_MAX_RESULTS = 10
LOCATION_SEARCH_RADIUS_KM = 5.0
TIME_WINDOW_DAYS = 30
RECENCY_BOOST_FACTOR = 0.2  # Boost factor for recent content
RELEVANCE_THRESHOLD = 0.65  # Minimum relevance score for results


class QueryFilter(BaseModel):
    """Model for query filtering parameters."""

    content_types: Optional[List[ContentType]] = None
    date_range: Optional[Dict[str, datetime]] = None
    location_filter: Optional[Dict[str, Any]] = None
    min_score: float = DEFAULT_MIN_SCORE
    max_results: int = DEFAULT_MAX_RESULTS


class SearchResult(BaseModel):
    """Model for combined search results."""

    vector_results: List[VectorSearchResult]
    structured_results: List[Dict[str, Any]]
    combined_score: float
    result_type: str  # "vector", "structured", "hybrid"


class QueryIntent(BaseModel):
    """Model for parsed query intent."""

    query_type: str  # "general", "location", "time", "media", "hybrid"
    location_terms: List[str]
    time_terms: List[str]
    media_terms: List[str]
    content_terms: List[str]
    has_location_filter: bool = False
    has_time_filter: bool = False
    has_media_filter: bool = False


class QueryProcessingService:
    """
    Service for processing natural language queries with vector similarity search,
    structured data filtering, and result ranking.
    """

    def __init__(self, embedding_service: EmbeddingService = None):
        """
        Initialize the query processing service.

        Args:
            embedding_service: Optional embedding service instance
        """
        self.embedding_service = embedding_service or embedding_service
        logger.info("Initialized QueryProcessingService")

    async def process_query(
        self, user: User, query_request: QueryRequest, db: AsyncSession
    ) -> QueryResponse:
        """
        Process a natural language query and return relevant results.

        Args:
            user: User making the query
            query_request: Query request data
            db: Database session

        Returns:
            QueryResponse with answer and sources

        Raises:
            HTTPException: If query processing fails
        """
        try:
            # Check usage limits
            if not await usage_service.can_make_query(user.id, db):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Daily query limit exceeded",
                )

            # Parse query intent
            intent = self._parse_query_intent(query_request.query)
            logger.debug(f"Parsed query intent: {intent.query_type}")

            # Create query filter based on intent and user subscription
            query_filter = await self._create_query_filter(user, intent, db)

            # Perform search based on intent
            search_results = await self._perform_search(
                user.id, query_request.query, intent, query_filter, db
            )

            # Rank and score results
            ranked_results = self._rank_and_score_results(
                search_results, intent, query_request.query
            )

            # Generate response
            response = await self._generate_response(
                user, query_request, ranked_results, db
            )

            # Update usage tracking
            await usage_service.increment_query_usage(user.id, db)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process query",
            )

    def _parse_query_intent(self, query: str) -> QueryIntent:
        """
        Parse natural language query to determine intent and extract key terms.

        Args:
            query: Natural language query

        Returns:
            QueryIntent with parsed information
        """
        query_lower = query.lower()

        # Location-related terms
        location_patterns = [
            r"\b(?:at|in|near|around|from|to)\s+([^,\s]+(?:\s+[^,\s]+)*)",
            r"\b(restaurant|cafe|store|shop|office|home|work|gym|park|beach|hotel)\b",
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:street|st|avenue|ave|road|rd|drive|dr|boulevard|blvd)\b",
        ]

        # Time-related terms
        time_patterns = [
            r"\b(?:yesterday|today|tomorrow)\b",
            r"\b(?:last|this|next)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
            r"\b(?:morning|afternoon|evening|night)\b",
            r"\b(?:ago|before|after|since|until|during)\b",
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b",
        ]

        # Media-related terms
        media_patterns = [
            r"\b(?:photo|picture|image|pic)\b",
            r"\b(?:voice|audio|recording|note)\b",
            r"\b(?:video|clip)\b",
        ]

        # Extract terms
        location_terms = []
        for pattern in location_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            location_terms.extend(
                [match if isinstance(match, str) else match[0] for match in matches]
            )

        time_terms = []
        for pattern in time_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            time_terms.extend(matches)

        media_terms = []
        for pattern in media_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            media_terms.extend(matches)

        # Extract content terms (remaining words after removing stop words)
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "under",
            "over",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "what",
            "where",
            "when",
            "why",
            "how",
            "who",
            "which",
            "that",
            "this",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
        }

        words = re.findall(r"\b\w+\b", query_lower)
        content_terms = [
            word for word in words if word not in stop_words and len(word) > 2
        ]

        # Determine query type
        query_type = "general"
        has_location = bool(location_terms)
        has_time = bool(time_terms)
        has_media = bool(media_terms)

        if has_location and has_time:
            query_type = "hybrid"
        elif has_location:
            query_type = "location"
        elif has_time:
            query_type = "time"
        elif has_media:
            query_type = "media"

        return QueryIntent(
            query_type=query_type,
            location_terms=location_terms,
            time_terms=time_terms,
            media_terms=media_terms,
            content_terms=content_terms,
            has_location_filter=has_location,
            has_time_filter=has_time,
            has_media_filter=has_media,
        )

    def _parse_time_range(self, time_terms: List[str]) -> Optional[Dict[str, datetime]]:
        """
        Parse time terms into a date range.

        Args:
            time_terms: List of time-related terms

        Returns:
            Dictionary with start_date and end_date, or None
        """
        if not time_terms:
            return None

        now = datetime.utcnow()
        start_date = None
        end_date = None

        for term in time_terms:
            term_lower = term.lower()

            # Handle relative terms
            if term_lower == "today":
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = now.replace(
                    hour=23, minute=59, second=59, microsecond=999999
                )
            elif term_lower == "yesterday":
                yesterday = now - timedelta(days=1)
                start_date = yesterday.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                end_date = yesterday.replace(
                    hour=23, minute=59, second=59, microsecond=999999
                )
            elif "last week" in term_lower:
                start_date = now - timedelta(days=7)
                end_date = now
            elif "last month" in term_lower:
                start_date = now - timedelta(days=30)
                end_date = now
            elif "last year" in term_lower:
                start_date = now - timedelta(days=365)
                end_date = now
            elif "this week" in term_lower:
                # Start of current week (Monday)
                days_since_monday = now.weekday()
                start_date = now - timedelta(days=days_since_monday)
                start_date = start_date.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                end_date = now
            elif "this month" in term_lower:
                start_date = now.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
                end_date = now

        if start_date or end_date:
            return {
                "start_date": start_date or (now - timedelta(days=365)),
                "end_date": end_date or now,
            }

        return None

    async def _create_query_filter(
        self, user: User, intent: QueryIntent, db: AsyncSession
    ) -> QueryFilter:
        """
        Create query filter based on intent and user subscription limits.

        Args:
            user: User making the query
            intent: Parsed query intent
            db: Database session

        Returns:
            QueryFilter with appropriate filters
        """
        query_filter = QueryFilter()

        # Apply subscription-based date range limits
        date_range = await usage_service.get_query_date_range(user.id, db)
        if date_range:
            query_filter.date_range = date_range

        # Apply content type filters based on media terms
        if intent.has_media_filter:
            content_types = []
            for term in intent.media_terms:
                if term in ["photo", "picture", "image", "pic"]:
                    content_types.extend(
                        [ContentType.IMAGE_TEXT, ContentType.IMAGE_DESC]
                    )
                elif term in ["voice", "audio", "recording"]:
                    content_types.append(ContentType.VOICE_TRANSCRIPT)

            if content_types:
                query_filter.content_types = content_types

        # Apply location filters
        if intent.has_location_filter:
            # For now, we'll use location terms for text-based filtering
            # In a more advanced implementation, we could geocode location terms
            query_filter.location_filter = {"names": intent.location_terms}

        # Apply time filters
        if intent.has_time_filter:
            time_range = self._parse_time_range(intent.time_terms)
            if time_range:
                if query_filter.date_range:
                    # Combine with subscription limits
                    start_date = max(
                        time_range.get(
                            "start_date", query_filter.date_range["start_date"]
                        ),
                        query_filter.date_range["start_date"],
                    )
                    end_date = min(
                        time_range.get("end_date", query_filter.date_range["end_date"]),
                        query_filter.date_range["end_date"],
                    )
                    query_filter.date_range = {
                        "start_date": start_date,
                        "end_date": end_date,
                    }
                else:
                    query_filter.date_range = time_range

        return query_filter

    async def _perform_search(
        self,
        user_id: UUID,
        query: str,
        intent: QueryIntent,
        query_filter: QueryFilter,
        db: AsyncSession,
    ) -> SearchResult:
        """
        Perform search based on query intent and filters.

        Args:
            user_id: User ID for isolation
            query: Original query text
            intent: Parsed query intent
            query_filter: Query filters
            db: Database session

        Returns:
            SearchResult with vector and structured results
        """
        vector_results = []
        structured_results = []

        # Adjust search parameters based on intent
        search_k = DEFAULT_VECTOR_SEARCH_K
        min_score = query_filter.min_score

        # For specific intents, we can adjust search parameters
        if intent.query_type == "location":
            # For location queries, we prioritize structured data but still want vector results
            search_k = int(DEFAULT_VECTOR_SEARCH_K * 0.7)  # Reduce vector results
            min_score = min_score + 0.05  # Increase minimum score threshold
        elif intent.query_type == "time":
            # For time queries, we want more results to filter by time later
            search_k = int(DEFAULT_VECTOR_SEARCH_K * 1.5)

        # Perform vector similarity search with user isolation
        try:
            # Ensure we have proper user isolation in the search
            vector_results = await self.embedding_service.search_by_text_async(
                user_id=user_id,  # User isolation is critical for privacy
                text=query,
                k=search_k,
                content_types=query_filter.content_types,
                min_score=min_score,
                date_range=query_filter.date_range,
                location_filter=query_filter.location_filter,
            )

            # Convert to VectorSearchResult models
            vector_results = self.embedding_service.convert_search_results_to_model(
                vector_results
            )

            # Log search metrics
            logger.debug(
                f"Vector search returned {len(vector_results)} results with avg score: "
                f"{sum(r.score for r in vector_results) / len(vector_results) if vector_results else 0}"
            )

        except Exception as e:
            logger.warning(f"Vector search failed: {str(e)}")
            vector_results = []

        # Perform structured data search based on intent
        if (
            intent.query_type in ["location", "time", "hybrid"]
            or intent.has_location_filter
            or intent.has_time_filter
        ):
            structured_results = await self._search_structured_data(
                user_id, intent, query_filter, db
            )
            logger.debug(
                f"Structured search returned {len(structured_results)} results"
            )

        # Determine result type
        result_type = "vector"
        if structured_results and not vector_results:
            result_type = "structured"
        elif structured_results and vector_results:
            result_type = "hybrid"

        # Calculate combined score with weighted approach
        combined_score = self._calculate_combined_score(
            vector_results, structured_results, intent
        )

        return SearchResult(
            vector_results=vector_results,
            structured_results=structured_results,
            combined_score=combined_score,
            result_type=result_type,
        )

    def _calculate_combined_score(
        self,
        vector_results: List[VectorSearchResult],
        structured_results: List[Dict[str, Any]],
        intent: QueryIntent,
    ) -> float:
        """
        Calculate a combined relevance score for search results.

        Args:
            vector_results: Vector search results
            structured_results: Structured search results
            intent: Query intent

        Returns:
            Combined relevance score
        """
        # Base weights for different result types
        vector_weight = 0.6
        structured_weight = 0.4

        # Adjust weights based on intent
        if intent.query_type == "location":
            vector_weight = 0.4
            structured_weight = 0.6
        elif intent.query_type == "time":
            vector_weight = 0.5
            structured_weight = 0.5

        # Calculate scores
        vector_score = 0.0
        if vector_results:
            # Use top 5 results for scoring
            top_results = sorted(vector_results, key=lambda x: x.score, reverse=True)[
                :5
            ]
            vector_score = sum(r.score for r in top_results) / len(top_results)

        structured_score = 0.0
        if structured_results:
            # Use top 5 results for scoring
            top_results = sorted(
                structured_results, key=lambda x: x.get("score", 0), reverse=True
            )[:5]
            structured_score = sum(r.get("score", 0) for r in top_results) / len(
                top_results
            )

        # Calculate weighted score
        if vector_results and structured_results:
            return (vector_score * vector_weight) + (
                structured_score * structured_weight
            )
        elif vector_results:
            return vector_score
        elif structured_results:
            return structured_score
        else:
            return 0.0

    async def _search_structured_data(
        self,
        user_id: UUID,
        intent: QueryIntent,
        query_filter: QueryFilter,
        db: AsyncSession,
    ) -> List[Dict[str, Any]]:
        """
        Search structured data based on query intent with advanced filtering.

        Args:
            user_id: User ID for isolation
            intent: Parsed query intent
            query_filter: Query filters
            db: Database session

        Returns:
            List of structured search results
        """
        results = []

        # Search location visits with enhanced filtering
        if intent.has_location_filter or intent.query_type in ["location", "hybrid"]:
            try:
                location_visits = []

                # Combine search strategies for better results
                # First try exact name matches
                if intent.location_terms:
                    location_visits = await location_visit_repository.search_by_names(
                        db, user_id, intent.location_terms, limit=10
                    )

                # Then try address search if no results
                if not location_visits and intent.location_terms:
                    for term in intent.location_terms:
                        if len(term) > 3:  # Only search with meaningful terms
                            visits = await location_visit_repository.search_by_address(
                                db, user_id, term, limit=5
                            )
                            location_visits.extend(visits)

                # Apply date range filtering if specified
                if query_filter.date_range:
                    # If we already have location visits, filter them by date
                    if location_visits:
                        filtered_visits = []
                        start_date = query_filter.date_range.get("start_date")
                        end_date = query_filter.date_range.get("end_date")

                        for visit in location_visits:
                            if (not start_date or visit.visit_time >= start_date) and (
                                not end_date or visit.visit_time <= end_date
                            ):
                                filtered_visits.append(visit)

                        location_visits = filtered_visits
                    # Otherwise, search directly with date range
                    else:
                        date_visits = (
                            await location_visit_repository.get_by_user_with_date_range(
                                db,
                                user_id,
                                start_date=query_filter.date_range.get("start_date"),
                                end_date=query_filter.date_range.get("end_date"),
                                limit=10,
                            )
                        )
                        location_visits.extend(date_visits)

                # Process results with dynamic scoring
                for visit in location_visits:
                    # Calculate base score - exact matches get higher scores
                    base_score = 0.85

                    # Boost score for name matches
                    if visit.names and intent.location_terms:
                        for term in intent.location_terms:
                            if any(
                                term.lower() in name.lower() for name in visit.names
                            ):
                                base_score += 0.1
                                break

                    # Boost score for recent visits
                    days_ago = (datetime.utcnow() - visit.visit_time).days
                    if days_ago <= 7:
                        base_score += 0.05

                    # Boost score for visits with descriptions
                    if visit.description and len(visit.description) > 10:
                        base_score += 0.05

                    # Cap score at 1.0
                    final_score = min(base_score, 1.0)

                    results.append(
                        {
                            "type": "location_visit",
                            "id": str(visit.id),
                            "content": f"Visited {visit.address or 'location'} on {visit.visit_time.strftime('%Y-%m-%d %H:%M')}",
                            "timestamp": visit.visit_time,
                            "location": {
                                "longitude": float(visit.longitude),
                                "latitude": float(visit.latitude),
                                "address": visit.address,
                                "names": visit.names,
                            },
                            "description": visit.description,
                            "score": final_score,
                        }
                    )

            except Exception as e:
                logger.warning(f"Location visit search failed: {str(e)}")

        # Search text notes with enhanced content matching
        if (
            intent.query_type in ["general", "hybrid"]
            or intent.has_location_filter
            or intent.has_time_filter
        ):
            try:
                text_notes = []
                combined_notes = set()  # Use set to avoid duplicates

                # Search by location names if location filter is present
                if intent.location_terms:
                    notes = await text_note_repository.search_by_names(
                        db, user_id, intent.location_terms, limit=5
                    )
                    for note in notes:
                        if str(note.id) not in combined_notes:
                            text_notes.append(note)
                            combined_notes.add(str(note.id))

                # Search by content terms for better semantic matching
                if intent.content_terms:
                    # Try combinations of terms for better matching
                    if len(intent.content_terms) > 1:
                        # Try pairs of terms
                        for i in range(len(intent.content_terms) - 1):
                            combined_term = (
                                f"{intent.content_terms[i]} {intent.content_terms[i+1]}"
                            )
                            notes = await text_note_repository.search_by_content(
                                db, user_id, combined_term, limit=3
                            )
                            for note in notes:
                                if str(note.id) not in combined_notes:
                                    text_notes.append(note)
                                    combined_notes.add(str(note.id))

                    # Try individual terms if we don't have enough results
                    if len(text_notes) < 5:
                        for term in intent.content_terms[:3]:  # Limit to first 3 terms
                            if len(term) > 3:  # Only search with meaningful terms
                                notes = await text_note_repository.search_by_content(
                                    db, user_id, term, limit=3
                                )
                                for note in notes:
                                    if str(note.id) not in combined_notes:
                                        text_notes.append(note)
                                        combined_notes.add(str(note.id))

                # Apply date range filtering if specified
                if query_filter.date_range and text_notes:
                    filtered_notes = []
                    start_date = query_filter.date_range.get("start_date")
                    end_date = query_filter.date_range.get("end_date")

                    for note in text_notes:
                        if (not start_date or note.timestamp >= start_date) and (
                            not end_date or note.timestamp <= end_date
                        ):
                            filtered_notes.append(note)

                    text_notes = filtered_notes

                # Process results with dynamic scoring
                for note in text_notes:
                    # Calculate base score
                    base_score = 0.8

                    # Boost score for content matches
                    if intent.content_terms:
                        content_lower = note.text_content.lower()
                        match_count = sum(
                            1
                            for term in intent.content_terms
                            if term.lower() in content_lower
                        )
                        if match_count > 0:
                            match_ratio = match_count / len(intent.content_terms)
                            base_score += min(
                                match_ratio * 0.15, 0.15
                            )  # Max 0.15 boost

                    # Boost score for location matches
                    if intent.location_terms and note.names:
                        for term in intent.location_terms:
                            if any(term.lower() in name.lower() for name in note.names):
                                base_score += 0.05
                                break

                    # Boost score for recent notes
                    days_ago = (datetime.utcnow() - note.timestamp).days
                    if days_ago <= 7:
                        base_score += 0.05

                    # Cap score at 1.0
                    final_score = min(base_score, 1.0)

                    results.append(
                        {
                            "type": "text_note",
                            "id": str(note.id),
                            "content": note.text_content[:150]
                            + ("..." if len(note.text_content) > 150 else ""),
                            "timestamp": note.timestamp,
                            "location": (
                                {
                                    "longitude": (
                                        float(note.longitude)
                                        if note.longitude
                                        else None
                                    ),
                                    "latitude": (
                                        float(note.latitude) if note.latitude else None
                                    ),
                                    "address": note.address,
                                    "names": note.names,
                                }
                                if note.longitude and note.latitude
                                else None
                            ),
                            "score": final_score,
                        }
                    )

            except Exception as e:
                logger.warning(f"Text note search failed: {str(e)}")

        # Search media files with enhanced filtering
        if intent.has_media_filter or intent.query_type == "media":
            try:
                media_files = []
                combined_media = set()  # Use set to avoid duplicates

                # Search by file type
                file_types = []
                for term in intent.media_terms:
                    if term in ["photo", "picture", "image", "pic"]:
                        file_types.append(FileType.PHOTO)
                    elif term in ["voice", "audio", "recording"]:
                        file_types.append(FileType.VOICE)

                # Search with date range if specified
                if query_filter.date_range:
                    for file_type in file_types or [None]:
                        media = await media_file_repository.get_by_user_with_date_range(
                            db,
                            user_id,
                            start_date=query_filter.date_range.get("start_date"),
                            end_date=query_filter.date_range.get("end_date"),
                            file_type=file_type,
                            limit=5,
                        )
                        for m in media:
                            if str(m.id) not in combined_media:
                                media_files.append(m)
                                combined_media.add(str(m.id))

                # Search by file type only if we don't have enough results
                if len(media_files) < 5:
                    for file_type in file_types or [None]:
                        media = await media_file_repository.get_by_file_type(
                            db, user_id, file_type=file_type, limit=5
                        )
                        for m in media:
                            if str(m.id) not in combined_media:
                                media_files.append(m)
                                combined_media.add(str(m.id))

                # Process results with dynamic scoring
                for media in media_files:
                    # Calculate base score
                    base_score = 0.8

                    # Boost score for file type matches
                    if media.file_type in file_types:
                        base_score += 0.05

                    # Boost score for recent media
                    days_ago = (datetime.utcnow() - media.timestamp).days
                    if days_ago <= 7:
                        base_score += 0.05

                    # Cap score at 1.0
                    final_score = min(base_score, 1.0)

                    results.append(
                        {
                            "type": "media_file",
                            "id": str(media.id),
                            "content": f"{media.file_type.title()} file: {media.original_filename or 'Unnamed_file'}",
                            "timestamp": media.timestamp,
                            "file_type": media.file_type,
                            "file_path": media.file_path,
                            "location": (
                                {
                                    "longitude": (
                                        float(media.longitude)
                                        if media.longitude
                                        else None
                                    ),
                                    "latitude": (
                                        float(media.latitude)
                                        if media.latitude
                                        else None
                                    ),
                                    "address": media.address,
                                    "names": media.names,
                                }
                                if media.longitude and media.latitude
                                else None
                            ),
                            "score": final_score,
                        }
                    )

            except Exception as e:
                logger.warning(f"Media file search failed: {str(e)}")

        return results

    def _rank_and_score_results(
        self, search_result: SearchResult, intent: QueryIntent, original_query: str
    ) -> List[Dict[str, Any]]:
        """
        Rank and score combined search results with advanced relevance scoring.

        Args:
            search_result: Combined search results
            intent: Parsed query intent
            original_query: Original query text

        Returns:
            List of ranked and scored results
        """
        all_results = []

        # Add vector results
        for result in search_result.vector_results:
            all_results.append(
                {
                    "type": result.content_type,
                    "id": result.content_id,
                    "content": result.content_text,
                    "timestamp": result.timestamp,
                    "source_id": str(result.source_id) if result.source_id else None,
                    "location": result.location,
                    "score": result.score,
                    "result_source": "vector_search",
                }
            )

        # Add structured results
        for result in search_result.structured_results:
            all_results.append({**result, "result_source": "structured_search"})

        # Apply advanced relevance scoring with multiple factors
        for result in all_results:
            # Start with base score
            base_score = result["score"]

            # Initialize boost factors
            intent_boost = 1.0
            recency_boost = 1.0
            content_match_boost = 1.0
            location_match_boost = 1.0

            # 1. Intent-based boosting
            if intent.query_type == "location":
                # For location queries, boost results with location data
                if result.get("location"):
                    intent_boost = 1.2
            elif intent.query_type == "time":
                # For time queries, boost recent results
                if result.get("timestamp"):
                    days_ago = (datetime.utcnow() - result["timestamp"]).days
                    if days_ago <= 3:
                        intent_boost = 1.3
                    elif days_ago <= 7:
                        intent_boost = 1.2
                    elif days_ago <= 30:
                        intent_boost = 1.1
            elif intent.query_type == "media":
                # For media queries, boost media file results
                if result.get("type") == "media_file":
                    intent_boost = 1.25
            elif intent.query_type == "hybrid":
                # For hybrid queries, boost results that match both location and time
                location_match = bool(result.get("location"))
                time_match = False

                # Check if timestamp falls within the time range
                time_range = self._parse_time_range(intent.time_terms)
                if time_range and result.get("timestamp"):
                    start_date = time_range.get("start_date")
                    end_date = time_range.get("end_date")
                    time_match = (
                        not start_date or result["timestamp"] >= start_date
                    ) and (not end_date or result["timestamp"] <= end_date)

                if location_match and time_match:
                    intent_boost = 1.5  # Strong boost for matching both aspects
                elif location_match or time_match:
                    intent_boost = 1.2  # Moderate boost for matching one aspect

            # 2. Recency boosting - newer content is generally more relevant
            if result.get("timestamp"):
                days_ago = (datetime.utcnow() - result["timestamp"]).days
                if days_ago <= 1:  # Today or yesterday
                    recency_boost = 1.3
                elif days_ago <= 7:  # Last week
                    recency_boost = 1.2
                elif days_ago <= 30:  # Last month
                    recency_boost = 1.1

            # 3. Content match boosting - exact phrase matches in content
            query_lower = original_query.lower()
            content_lower = result.get("content", "").lower()

            if query_lower in content_lower:
                content_match_boost = 1.4  # Strong boost for exact phrase match
            else:
                # Check for term matches
                matched_terms = 0
                for term in intent.content_terms:
                    if term.lower() in content_lower:
                        matched_terms += 1

                # Boost based on proportion of matched terms
                if intent.content_terms:
                    match_ratio = matched_terms / len(intent.content_terms)
                    if match_ratio > 0.75:
                        content_match_boost = 1.3
                    elif match_ratio > 0.5:
                        content_match_boost = 1.2
                    elif match_ratio > 0.25:
                        content_match_boost = 1.1

            # 4. Location match boosting - match with location data
            if result.get("location") and intent.location_terms:
                location_data = result["location"]
                address = location_data.get("address", "").lower()
                names = location_data.get("names", [])

                # Check for matches in address or names
                address_match = (
                    any(term.lower() in address for term in intent.location_terms)
                    if address
                    else False
                )
                names_match = (
                    any(
                        any(term.lower() in name.lower() for name in names)
                        for term in intent.location_terms
                    )
                    if names
                    else False
                )

                if address_match and names_match:
                    location_match_boost = 1.3
                elif address_match or names_match:
                    location_match_boost = 1.2

            # Calculate boost factor with a weighted combination
            boost_factor = (
                intent_boost * 0.4
                + recency_boost * 0.3
                + content_match_boost * 0.2
                + location_match_boost * 0.1
            )

            # Add relevance factors for debugging
            result["relevance_factors"] = {
                "base_score": base_score,
                "intent_boost": intent_boost,
                "recency_boost": recency_boost,
                "content_match_boost": content_match_boost,
                "location_match_boost": location_match_boost,
                "final_boost": boost_factor,
            }

            # Apply boost to score with a cap
            result["score"] = min(base_score * boost_factor, 1.0)

        # Filter out low-relevance results
        filtered_results = [r for r in all_results if r["score"] >= RELEVANCE_THRESHOLD]

        # If we filtered too aggressively, fall back to original results
        if not filtered_results and all_results:
            filtered_results = all_results

        # Sort by score (descending) and timestamp (descending)
        filtered_results.sort(
            key=lambda x: (x.get("score", 0), x.get("timestamp", datetime.min)),
            reverse=True,
        )

        # Limit results
        return filtered_results[:DEFAULT_MAX_RESULTS]

    async def _generate_response(
        self,
        user: User,
        query_request: QueryRequest,
        ranked_results: List[Dict[str, Any]],
        db: AsyncSession,
    ) -> QueryResponse:
        """
        Generate a natural language response from search results.

        Args:
            user: User making the query
            query_request: Query request data
            ranked_results: Ranked search results
            db: Database session

        Returns:
            QueryResponse with answer and sources
        """
        # Prepare sources and media references
        sources = []
        media_references = []

        # If no results, return a default message
        if not ranked_results:
            # Create or update session
            session_id = query_request.session_id

            if session_id:
                try:
                    session = await query_session_repository.get_by_user(
                        db, user.id, UUID(session_id)
                    )
                except (ValueError, TypeError):
                    session = await query_session_repository.create(db, user.id, {})
            else:
                session = await query_session_repository.create(db, user.id, {})

            return QueryResponse(
                answer="I couldn't find any relevant information for your query. Try asking about your location visits, notes, or uploaded photos and voice recordings.",
                sources=[],
                media_references=None,
                session_id=str(session.id),
            )

        # Process results for context
        context_items = []
        result_types = set()

        for result in ranked_results:
            result_type = result["type"]
            result_types.add(result_type)

            if result_type == "location_visit":
                context_items.append(
                    {
                        "type": "location_visit",
                        "content": result.get("content", ""),
                        "description": result.get("description", ""),
                        "timestamp": result.get("timestamp"),
                        "location": result.get("location"),
                    }
                )

            elif result_type == "text_note":
                context_items.append(
                    {
                        "type": "text_note",
                        "content": result.get("content", ""),
                        "timestamp": result.get("timestamp"),
                        "location": result.get("location"),
                    }
                )

            elif result_type == "media_file":
                context_items.append(
                    {
                        "type": "media_file",
                        "content": result.get("content", ""),
                        "file_type": result.get("file_type", ""),
                        "timestamp": result.get("timestamp"),
                        "location": result.get("location"),
                    }
                )

                # Add media reference
                try:
                    media_ref = MediaReference(
                        media_id=UUID(result["id"]),
                        media_type=result["file_type"],
                        description=result.get("content", "Media file"),
                        timestamp=result["timestamp"],
                        location=result.get("location"),
                    )
                    media_references.append(media_ref)
                except (ValueError, KeyError):
                    pass

            # Add to sources
            source = {
                "type": result["type"],
                "content": result.get("content", ""),
                "timestamp": result.get("timestamp", ""),
                "score": result.get("score", 0.0),
                "location": result.get("location"),
            }
            sources.append(source)

        # Generate natural language response
        result_count = len(ranked_results)

        # Group results by type
        location_visits = [r for r in ranked_results if r["type"] == "location_visit"]
        text_notes = [r for r in ranked_results if r["type"] == "text_note"]
        media_files = [r for r in ranked_results if r["type"] == "media_file"]

        # Start with a basic response
        if result_count == 1:
            answer = f"I found 1 relevant item"
        else:
            answer = f"I found {result_count} relevant items"

        # Add details about top result
        top_result = ranked_results[0]

        # Format timestamp if available
        timestamp_str = ""
        try:
            if top_result.get("timestamp"):
                timestamp = top_result["timestamp"]
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                timestamp_str = timestamp.strftime("%B %d, %Y")
        except (ValueError, TypeError):
            pass

        # Add top result details based on type
        if top_result["type"] == "location_visit":
            location_name = ""
            if top_result["location"] and top_result["location"].get("names"):
                location_name = top_result["location"]["names"][0]
            elif top_result["location"] and top_result["location"].get("address"):
                location_name = top_result["location"]["address"]

            answer += f" The most relevant result is a visit to {location_name}"
            if timestamp_str:
                answer += f" on {timestamp_str}"

            if top_result.get("description"):
                answer += f". {top_result['description']}"

        elif top_result["type"] == "text_note":
            answer += f" The most relevant result is a text note"
            if timestamp_str:
                answer += f" from {timestamp_str}"

            # Add a snippet of the content
            content = ""
            if top_result.get("content"):
                content = top_result["content"]
                if len(content) > 100:
                    content = content[:97] + "..."

            if content:
                answer += f': "{content}"'

        elif top_result["type"] == "media_file":
            media_type = top_result.get("file_type", "media file").lower()
            answer += f" The most relevant result is a {media_type}"
            if timestamp_str:
                answer += f" from {timestamp_str}"

            # Add location context if available
            if top_result["location"] and top_result["location"].get("address"):
                answer += f" taken at {top_result['location']['address']}"

        # Add summary of other types of findings
        if result_count > 1:
            # Find unique locations
            locations = set()
            for visit in location_visits:
                if visit.get("location"):
                    if visit["location"].get("names"):
                        locations.add(visit["location"]["names"][0])
                    elif visit["location"].get("address"):
                        locations.add(visit["location"]["address"])

            # Find unique dates
            dates = []
            for item in context_items[:3]:  # Look at top 3 results
                if item.get("timestamp"):
                    try:
                        timestamp = item["timestamp"]
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00")
                            )
                        dates.append(timestamp.strftime("%B %d, %Y"))
                    except (ValueError, TypeError):
                        pass

            unique_dates = list(set(dates))

            # For time-based queries, emphasize the timestamps
            if (
                "when" in query_request.query.lower()
                or "time" in query_request.query.lower()
                or "date" in query_request.query.lower()
            ):
                if dates:
                    if len(unique_dates) == 1:
                        answer += f" This occurred on {unique_dates[0]}."
                    elif len(unique_dates) == 2:
                        answer += f" These events occurred on {unique_dates[0]} and {unique_dates[1]}."
                    else:
                        answer += f" These events occurred on various dates including {', '.join(unique_dates[:3])}."

            # For location-based queries, emphasize the locations
            elif (
                "where" in query_request.query.lower()
                or "location" in query_request.query.lower()
                or "place" in query_request.query.lower()
            ):
                if locations:
                    if len(list(locations)) == 1:
                        answer += f" This occurred at {list(locations)[0]}."
                    elif len(list(locations)) == 2:
                        answer += f" These events occurred at {list(locations)[0]} and {list(locations)[1]}."
                    else:
                        answer += f" These events occurred at various locations including {', '.join(list(locations)[:3])}."

            # Add additional info summary
            additional_info = []

            # Count by media type
            photos = sum(1 for m in media_files if m.get("file_type") == FileType.PHOTO)
            voice = sum(1 for m in media_files if m.get("file_type") == FileType.VOICE)

            if photos and voice:
                additional_info.append(f"{photos} photos and {voice} voice recordings")
            elif photos:
                additional_info.append(f"{photos} photos")
            elif voice:
                additional_info.append(f"{voice} voice recordings")

            # Add text notes count
            if len(text_notes) > 1:
                additional_info.append(f"{len(text_notes)} text notes")

            # Add location visits
            if len(location_visits) > 1:
                location_list = list(locations)[:3]  # Limit to 3 locations
                if len(location_list) > 1:
                    additional_info.append(
                        f"visits to {', '.join(location_list[:-1])} and {location_list[-1]}"
                    )
                elif location_list:
                    additional_info.append(f"visit to {location_list[0]}")

            if additional_info:
                answer += f" I also found {', '.join(additional_info)}."

        # Add note about media files if present
        if media_files:
            answer += " You can view the referenced media files."

        # Handle session management
        session = None
        session_id = query_request.session_id

        if session_id:
            # Use existing session
            try:
                session = await query_session_repository.get_by_user(
                    db, user.id, UUID(session_id)
                )
            except (ValueError, TypeError):
                session_id = None

        if not session:
            # Create new session
            session = await query_session_repository.create(
                db, user_id=user.id, context={}
            )
            session_id = str(session.id)

        # Update session context
        context = session.session_context or {}
        context["last_query"] = query_request.query
        context["query_count"] = context.get("query_count", 0) + 1
        context["last_results"] = [r.get("id") for r in ranked_results[:5]]

        # Update media references in context
        if "media_references" not in context:
            context["media_references"] = []

        for media_ref in media_references:
            context["media_references"].append(
                {
                    "media_id": str(media_ref.media_id),
                    "media_type": media_ref.media_type,
                    "description": media_ref.description,
                    "timestamp": media_ref.timestamp.isoformat(),
                }
            )

        # Keep only the most recent media references
        context["media_references"] = context["media_references"][-10:]

        # Update session
        await query_session_repository.update_context(db, session.id, context)

        return QueryResponse(
            answer=answer,
            sources=sources,
            media_references=media_references if media_references else None,
            session_id=session_id,
        )


# Create a singleton instance
query_processing_service = QueryProcessingService(embedding_service=embedding_service)
