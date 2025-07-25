"""
LLM service for ChronoTrail API.

This module provides functionality for generating natural language responses
using Amazon Bedrock LLMs, with proper context assembly and error handling.
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from fastapi import HTTPException, status

from app.aws.clients import (
    get_async_bedrock_client,
    get_bedrock_client,
    handle_aws_error,
    handle_aws_error_async,
    with_retry,
)
from app.aws.exceptions import BedrockError
from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import FileType

# Configure logger
logger = get_logger(__name__)

# Constants for LLM
DEFAULT_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"  # Claude 3 Sonnet
FALLBACK_MODEL_ID = "anthropic.claude-instant-v1"  # Claude Instant as fallback
MAX_TOKENS = 1024  # Maximum tokens for response
TEMPERATURE = 0.7  # Temperature for response generation (0.0-1.0)
TOP_P = 0.9  # Top-p sampling parameter (0.0-1.0)


class LLMService:
    """
    Service for generating natural language responses using Amazon Bedrock LLMs.

    This service provides methods for generating responses based on search results
    and user queries, with proper context assembly and error handling.
    """

    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize the LLM service.

        Args:
            model_id: Optional model ID (defaults to DEFAULT_MODEL_ID)
        """
        self.model_id = model_id or DEFAULT_MODEL_ID
        logger.info(f"Initialized LLMService with model: {self.model_id}")

    def _format_context_for_claude(self, context_items: List[Dict[str, Any]]) -> str:
        """
        Format context items for Claude prompt.

        Args:
            context_items: List of context items from search results

        Returns:
            str: Formatted context for Claude prompt
        """
        if not context_items:
            return "No relevant information found."

        formatted_context = []

        for i, item in enumerate(context_items, 1):
            item_type = item.get("type", "unknown")
            content = item.get("content", "")
            timestamp = item.get("timestamp", "")

            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
                    timestamp_str = timestamp.strftime("%B %d, %Y")
                except (ValueError, TypeError):
                    timestamp_str = "unknown date"
            else:
                timestamp_str = "unknown date"

            # Format based on item type
            if item_type == "location_visit":
                location_info = item.get("location", {})
                address = location_info.get("address", "unknown location")
                description = item.get("description", "")

                entry = f"{i}. Location Visit on {timestamp_str}: {address}"
                if description:
                    entry += f". {description}"

                formatted_context.append(entry)

            elif item_type == "text_note":
                entry = f"{i}. Text Note from {timestamp_str}: {content}"
                formatted_context.append(entry)

            elif item_type == "media_file":
                file_type = item.get("file_type", "")
                file_type_str = (
                    "Photo" if file_type == FileType.PHOTO else "Voice Recording"
                )

                entry = f"{i}. {file_type_str} from {timestamp_str}: {content}"
                formatted_context.append(entry)

            else:
                entry = f"{i}. {item_type.capitalize()} from {timestamp_str}: {content}"
                formatted_context.append(entry)

        return "\n".join(formatted_context)

    def _create_claude_prompt(
        self,
        query: str,
        context_items: List[Dict[str, Any]],
        media_references: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Create a prompt for Claude.

        Args:
            query: User's query
            context_items: Context items from search results
            media_references: Optional media references

        Returns:
            str: Formatted prompt for Claude
        """
        formatted_context = self._format_context_for_claude(context_items)

        # Create the prompt
        prompt = f"""<instructions>
You are an AI assistant for ChronoTrail, a personal timeline app. Your task is to answer the user's question based ONLY on the provided context information. The context contains information from the user's personal timeline, including location visits, text notes, photos, and voice recordings.

Guidelines:
1. Answer ONLY based on the provided context. Do not make up information.
2. If the context doesn't contain relevant information to answer the question, say so clearly.
3. Be concise and direct in your answers.
4. When referring to photos or voice recordings, mention them specifically so the user knows they can view/listen to them.
5. Format dates in a natural, readable way.
6. Do not mention that you are an AI or that you're using provided context - just answer naturally.
7. Keep your response under 150 words unless more detail is necessary.
</instructions>

<context>
{formatted_context}
</context>

<question>
{query}
</question>

<answer>
"""

        return prompt

    def _create_anthropic_payload(
        self,
        prompt: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ) -> Dict[str, Any]:
        """
        Create a payload for Anthropic Claude models.

        Args:
            prompt: Formatted prompt
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
            top_p: Top-p sampling parameter

        Returns:
            Dict[str, Any]: Payload for Anthropic Claude models
        """
        return {
            "prompt": prompt,
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": ["</answer>"],
        }

    def _create_amazon_payload(
        self,
        prompt: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ) -> Dict[str, Any]:
        """
        Create a payload for Amazon Titan models.

        Args:
            prompt: Formatted prompt
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
            top_p: Top-p sampling parameter

        Returns:
            Dict[str, Any]: Payload for Amazon Titan models
        """
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": top_p,
                "stopSequences": ["</answer>"],
            },
        }

    @handle_aws_error(service_name="bedrock-runtime")
    def generate_response(
        self,
        query: str,
        context_items: List[Dict[str, Any]],
        media_references: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate a natural language response using Amazon Bedrock.

        Args:
            query: User's query
            context_items: Context items from search results
            media_references: Optional media references

        Returns:
            str: Generated response

        Raises:
            BedrockError: If response generation fails
        """
        try:
            # Get Bedrock client
            bedrock_client = get_bedrock_client()

            # Create prompt
            prompt = self._create_claude_prompt(query, context_items, media_references)

            # Prepare request body based on model
            if "anthropic" in self.model_id.lower():
                request_body = json.dumps(self._create_anthropic_payload(prompt))
            elif "amazon" in self.model_id.lower():
                request_body = json.dumps(self._create_amazon_payload(prompt))
            else:
                raise BedrockError(
                    message=f"Unsupported model: {self.model_id}",
                    operation="generate_response",
                )

            # Call Bedrock API
            response = bedrock_client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=request_body,
            )

            # Parse response
            response_body = json.loads(response.get("body").read())

            # Extract response text based on model
            if "anthropic" in self.model_id.lower():
                response_text = response_body.get("completion", "")
            elif "amazon" in self.model_id.lower():
                response_text = response_body.get("results", [{}])[0].get(
                    "outputText", ""
                )
            else:
                response_text = ""

            if not response_text:
                raise BedrockError(
                    message="Empty response from Bedrock", operation="generate_response"
                )

            # Clean up response
            response_text = response_text.strip()

            return response_text

        except BedrockError:
            # Re-raise BedrockError
            raise
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")

            # Try fallback model if different from current model
            if self.model_id != FALLBACK_MODEL_ID:
                logger.info(f"Trying fallback model: {FALLBACK_MODEL_ID}")
                try:
                    # Create a temporary LLMService with fallback model
                    fallback_service = LLMService(model_id=FALLBACK_MODEL_ID)
                    return fallback_service.generate_response(
                        query, context_items, media_references
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback model failed: {str(fallback_error)}")

            # If all fails, return a generic response
            return self._generate_fallback_response(query, context_items)

    @handle_aws_error_async(service_name="bedrock-runtime")
    async def generate_response_async(
        self,
        query: str,
        context_items: List[Dict[str, Any]],
        media_references: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate a natural language response using Amazon Bedrock (async).

        Args:
            query: User's query
            context_items: Context items from search results
            media_references: Optional media references

        Returns:
            str: Generated response

        Raises:
            BedrockError: If response generation fails
        """
        try:
            # Get async Bedrock client factory
            get_client = get_async_bedrock_client()

            # Create prompt
            prompt = self._create_claude_prompt(query, context_items, media_references)

            # Prepare request body based on model
            if "anthropic" in self.model_id.lower():
                request_body = json.dumps(self._create_anthropic_payload(prompt))
            elif "amazon" in self.model_id.lower():
                request_body = json.dumps(self._create_amazon_payload(prompt))
            else:
                raise BedrockError(
                    message=f"Unsupported model: {self.model_id}",
                    operation="generate_response_async",
                )

            # Use async context manager to get client
            async with get_client() as bedrock_client:
                # Call Bedrock API
                response = await bedrock_client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=request_body,
                )

                # Parse response
                response_body = json.loads(await response.get("body").read())

                # Extract response text based on model
                if "anthropic" in self.model_id.lower():
                    response_text = response_body.get("completion", "")
                elif "amazon" in self.model_id.lower():
                    response_text = response_body.get("results", [{}])[0].get(
                        "outputText", ""
                    )
                else:
                    response_text = ""

                if not response_text:
                    raise BedrockError(
                        message="Empty response from Bedrock",
                        operation="generate_response_async",
                    )

                # Clean up response
                response_text = response_text.strip()

                return response_text

        except BedrockError:
            # Re-raise BedrockError
            raise
        except Exception as e:
            logger.error(f"Failed to generate async response: {str(e)}")

            # For async, just use the fallback response directly
            return self._generate_fallback_response(query, context_items)

    def _generate_fallback_response(
        self, query: str, context_items: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a fallback response when LLM fails.

        Args:
            query: User's query
            context_items: Context items from search results

        Returns:
            str: Fallback response
        """
        # Count items by type
        location_visits = [
            item for item in context_items if item.get("type") == "location_visit"
        ]
        text_notes = [item for item in context_items if item.get("type") == "text_note"]
        media_files = [
            item for item in context_items if item.get("type") == "media_file"
        ]

        # Generate a simple response based on available data
        if not context_items:
            return "I couldn't find any relevant information for your query. Try asking about your location visits, notes, or uploaded photos and voice recordings."

        result_count = len(context_items)
        response = f"I found {result_count} relevant items"

        # Add type information
        type_descriptions = []
        if location_visits:
            type_descriptions.append(f"{len(location_visits)} location visits")
        if text_notes:
            type_descriptions.append(f"{len(text_notes)} text notes")
        if media_files:
            photos = sum(1 for m in media_files if m.get("file_type") == FileType.PHOTO)
            voice = sum(1 for m in media_files if m.get("file_type") == FileType.VOICE)

            if photos and voice:
                type_descriptions.append(
                    f"{photos} photos and {voice} voice recordings"
                )
            elif photos:
                type_descriptions.append(f"{photos} photos")
            elif voice:
                type_descriptions.append(f"{voice} voice recordings")

        if type_descriptions:
            response += f" including {', '.join(type_descriptions)}"

        # Add time information for time-based queries
        if any(term in query.lower() for term in ["when", "time", "date"]):
            dates = []
            for item in context_items[:3]:  # Look at top 3 results
                if item.get("timestamp"):
                    try:
                        timestamp = (
                            datetime.fromisoformat(
                                item["timestamp"].replace("Z", "+00:00")
                            )
                            if isinstance(item["timestamp"], str)
                            else item["timestamp"]
                        )
                        dates.append(timestamp.strftime("%B %d, %Y"))
                    except (ValueError, TypeError):
                        pass

            unique_dates = list(set(dates))
            if unique_dates:
                if len(unique_dates) == 1:
                    response += f" This occurred on {unique_dates[0]}."
                elif len(unique_dates) == 2:
                    response += f" These events occurred on {unique_dates[0]} and {unique_dates[1]}."
                else:
                    response += f" These events occurred on various dates including {', '.join(unique_dates[:3])}."

        # Add location information for location-based queries
        if any(term in query.lower() for term in ["where", "location", "place"]):
            locations = []
            for item in context_items[:3]:  # Look at top 3 results
                loc = item.get("location")
                if loc:
                    if loc.get("address"):
                        locations.append(loc["address"])
                    elif loc.get("names"):
                        locations.append(loc["names"][0])

            unique_locations = list(set(locations))
            if unique_locations:
                if len(unique_locations) == 1:
                    response += f" This occurred at {unique_locations[0]}."
                elif len(unique_locations) == 2:
                    response += f" These events occurred at {unique_locations[0]} and {unique_locations[1]}."
                else:
                    response += f" These events occurred at various locations including {', '.join(unique_locations[:3])}."

        # Add a note about media files if present
        if media_files:
            response += " You can view the referenced media files."

        return response


# Create a singleton instance
llm_service = LLMService()
