"""
OCR and image analysis service for ChronoTrail API.

This module provides functionality for extracting text from images using Amazon Textract
and generating image descriptions using Amazon Rekognition, with proper error handling
and fallback mechanisms.
"""
import io
import base64
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from uuid import UUID

from fastapi import UploadFile, HTTPException, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger
from app.aws.clients import (
    get_textract_client,
    get_rekognition_client,
    get_async_textract_client,
    get_async_rekognition_client,
    handle_aws_error,
    handle_aws_error_async,
)
from app.aws.exceptions import TextractError, RekognitionError, FileProcessingError
from app.models.schemas import ProcessedPhoto

# Configure logger
logger = get_logger(__name__)

# Maximum image size for direct API processing (5MB for Textract, 5MB for Rekognition)
MAX_DIRECT_PROCESSING_SIZE = 5 * 1024 * 1024


class ImageProcessingService:
    """
    Service for OCR and image analysis using AWS services.
    
    This service provides methods for extracting text from images using Amazon Textract
    and generating image descriptions using Amazon Rekognition, with proper error handling
    and fallback mechanisms.
    """
    
    def __init__(self):
        """Initialize the image processing service."""
        logger.info("Initialized ImageProcessingService")
    
    @handle_aws_error(service_name='textract')
    def extract_text_from_image_sync(self, image_data: bytes) -> str:
        """
        Extract text from an image using Amazon Textract (synchronous).
        
        Args:
            image_data: Raw image data as bytes
            
        Returns:
            str: Extracted text
            
        Raises:
            TextractError: If text extraction fails
        """
        try:
            # Get Textract client
            textract_client = get_textract_client()
            
            # Call Textract API
            response = textract_client.detect_document_text(
                Document={'Bytes': image_data}
            )
            
            # Extract text from response
            extracted_text = ""
            for item in response.get('Blocks', []):
                if item.get('BlockType') == 'LINE':
                    extracted_text += item.get('Text', '') + "\n"
            
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Failed to extract text from image: {str(e)}")
            raise TextractError(
                message=f"Failed to extract text from image: {str(e)}",
                operation="extract_text_from_image_sync"
            )
    
    @handle_aws_error_async(service_name='textract')
    async def extract_text_from_image(self, image_data: bytes) -> str:
        """
        Extract text from an image using Amazon Textract (asynchronous).
        
        Args:
            image_data: Raw image data as bytes
            
        Returns:
            str: Extracted text
            
        Raises:
            TextractError: If text extraction fails
        """
        try:
            # Get async Textract client
            textract_client_factory = get_async_textract_client()
            
            # Call Textract API
            async with textract_client_factory() as textract_client:
                response = await textract_client.detect_document_text(
                    Document={'Bytes': image_data}
                )
            
            # Extract text from response
            extracted_text = ""
            for item in response.get('Blocks', []):
                if item.get('BlockType') == 'LINE':
                    extracted_text += item.get('Text', '') + "\n"
            
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Failed to extract text from image: {str(e)}")
            raise TextractError(
                message=f"Failed to extract text from image: {str(e)}",
                operation="extract_text_from_image"
            )
    
    @handle_aws_error(service_name='rekognition')
    def describe_image_sync(self, image_data: bytes) -> Dict[str, Any]:
        """
        Generate a description of an image using Amazon Rekognition (synchronous).
        
        Args:
            image_data: Raw image data as bytes
            
        Returns:
            Dict: Image analysis results including labels, objects, and text
            
        Raises:
            RekognitionError: If image analysis fails
        """
        try:
            # Get Rekognition client
            rekognition_client = get_rekognition_client()
            
            # Call Rekognition API for label detection
            label_response = rekognition_client.detect_labels(
                Image={'Bytes': image_data},
                MaxLabels=10,
                MinConfidence=70
            )
            
            # Extract labels
            labels = [
                {
                    'name': label.get('Name', ''),
                    'confidence': label.get('Confidence', 0),
                    'parents': [parent.get('Name', '') for parent in label.get('Parents', [])]
                }
                for label in label_response.get('Labels', [])
            ]
            
            # Call Rekognition API for text detection
            text_response = rekognition_client.detect_text(
                Image={'Bytes': image_data}
            )
            
            # Extract detected text
            detected_text = " ".join([
                text.get('DetectedText', '')
                for text in text_response.get('TextDetections', [])
                if text.get('Type') == 'LINE'
            ])
            
            # Generate a simple description based on labels
            label_names = [label['name'] for label in labels]
            description = f"Image contains: {', '.join(label_names)}"
            
            return {
                'labels': labels,
                'detected_text': detected_text,
                'description': description
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze image: {str(e)}")
            raise RekognitionError(
                message=f"Failed to analyze image: {str(e)}",
                operation="describe_image_sync"
            )
    
    @handle_aws_error_async(service_name='rekognition')
    async def describe_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Generate a description of an image using Amazon Rekognition (asynchronous).
        
        Args:
            image_data: Raw image data as bytes
            
        Returns:
            Dict: Image analysis results including labels, objects, and text
            
        Raises:
            RekognitionError: If image analysis fails
        """
        try:
            # Get async Rekognition client
            rekognition_client_factory = get_async_rekognition_client()
            
            # Call Rekognition API for label detection
            async with rekognition_client_factory() as rekognition_client:
                label_response = await rekognition_client.detect_labels(
                    Image={'Bytes': image_data},
                    MaxLabels=10,
                    MinConfidence=70
                )
            
            # Extract labels
            labels = [
                {
                    'name': label.get('Name', ''),
                    'confidence': label.get('Confidence', 0),
                    'parents': [parent.get('Name', '') for parent in label.get('Parents', [])]
                }
                for label in label_response.get('Labels', [])
            ]
            
            # Call Rekognition API for text detection
            async with rekognition_client_factory() as rekognition_client:
                text_response = await rekognition_client.detect_text(
                    Image={'Bytes': image_data}
                )
            
            # Extract detected text
            detected_text = " ".join([
                text.get('DetectedText', '')
                for text in text_response.get('TextDetections', [])
                if text.get('Type') == 'LINE'
            ])
            
            # Generate a simple description based on labels
            label_names = [label['name'] for label in labels]
            description = f"Image contains: {', '.join(label_names)}"
            
            return {
                'labels': labels,
                'detected_text': detected_text,
                'description': description
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze image: {str(e)}")
            raise RekognitionError(
                message=f"Failed to analyze image: {str(e)}",
                operation="describe_image"
            )
    
    async def process_image(
        self,
        file: Union[UploadFile, bytes],
        original_id: Optional[UUID] = None
    ) -> ProcessedPhoto:
        """
        Process an image using OCR and image analysis with fallback mechanisms.
        
        This method attempts to extract text and generate a description for an image.
        If one service fails, it will still return partial results from the other service.
        
        Args:
            file: Image file as UploadFile or raw bytes
            original_id: Optional ID of the original file record
            
        Returns:
            ProcessedPhoto: Processing results
            
        Raises:
            FileProcessingError: If both text extraction and image analysis fail
        """
        # Read file content if UploadFile
        if isinstance(file, UploadFile):
            content = await file.read()
            await file.seek(0)  # Reset file pointer for potential future use
        else:
            content = file
        
        # Initialize result
        result = ProcessedPhoto(
            original_id=original_id or UUID('00000000-0000-0000-0000-000000000000'),
            processed_text="",
            content_type="image_text",
            processing_status="completed",
            extracted_text=None,
            image_description=None,
            detected_objects=None
        )
        
        # Track errors for fallback handling
        textract_error = None
        rekognition_error = None
        
        # Try to extract text using Textract
        try:
            extracted_text = await self.extract_text_from_image(content)
            result.extracted_text = extracted_text
            result.processed_text += extracted_text
        except TextractError as e:
            logger.warning(f"Textract processing failed: {str(e)}")
            textract_error = str(e)
            result.processing_status = "partial"
        
        # Try to analyze image using Rekognition
        try:
            image_analysis = await self.describe_image(content)
            result.image_description = image_analysis.get('description', '')
            result.detected_objects = [label['name'] for label in image_analysis.get('labels', [])]
            
            # Add image description to processed text if no text was extracted
            if not result.extracted_text or result.extracted_text.strip() == "":
                result.processed_text += image_analysis.get('description', '')
                result.content_type = "image_desc"
            
            # Add detected text from Rekognition if Textract failed
            if textract_error and image_analysis.get('detected_text'):
                result.extracted_text = image_analysis.get('detected_text', '')
                result.processed_text += "\n" + image_analysis.get('detected_text', '')
                
        except RekognitionError as e:
            logger.warning(f"Rekognition processing failed: {str(e)}")
            rekognition_error = str(e)
            result.processing_status = "partial"
        
        # Handle case where both services failed
        if textract_error and rekognition_error:
            result.processing_status = "failed"
            result.error_details = f"Text extraction: {textract_error}; Image analysis: {rekognition_error}"
            result.processed_text = "Image processing failed"
            
            logger.error(f"Image processing failed completely: {result.error_details}")
            
            # Still return partial result rather than raising exception
            # to allow for graceful degradation
        
        # Handle case where no meaningful content was extracted
        if result.processed_text.strip() == "" or result.processed_text.strip() == "Image contains:":
            result.processed_text = "No text or recognizable content found in image"
            result.processing_status = "partial"
        
        return result


# Create a singleton instance
image_processing_service = ImageProcessingService()