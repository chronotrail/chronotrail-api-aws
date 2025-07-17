"""
Tests for OCR and image analysis service.
"""
import io
import uuid
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi import UploadFile
from botocore.exceptions import ClientError

from app.aws.image_processing import ImageProcessingService
from app.aws.exceptions import TextractError, RekognitionError, FileProcessingError
from app.models.schemas import ProcessedPhoto


@pytest.fixture
def image_processing_service():
    """Create an ImageProcessingService instance for testing."""
    return ImageProcessingService()


@pytest.fixture
def mock_image_file():
    """Create a mock image file for testing."""
    file_content = b"test image content"
    file = MagicMock(spec=UploadFile)
    file.filename = "test.jpg"
    file.content_type = "image/jpeg"
    file.file = io.BytesIO(file_content)
    file.read = AsyncMock(return_value=file_content)
    file.seek = AsyncMock()
    return file


@pytest.fixture
def mock_textract_response():
    """Create a mock Textract response."""
    return {
        'Blocks': [
            {
                'BlockType': 'LINE',
                'Text': 'This is a test',
                'Confidence': 99.5
            },
            {
                'BlockType': 'LINE',
                'Text': 'OCR extraction',
                'Confidence': 98.7
            }
        ]
    }


@pytest.fixture
def mock_rekognition_label_response():
    """Create a mock Rekognition label detection response."""
    return {
        'Labels': [
            {
                'Name': 'Person',
                'Confidence': 99.2,
                'Parents': []
            },
            {
                'Name': 'Mountain',
                'Confidence': 95.8,
                'Parents': [{'Name': 'Nature'}]
            },
            {
                'Name': 'Tree',
                'Confidence': 94.3,
                'Parents': [{'Name': 'Plant'}, {'Name': 'Nature'}]
            }
        ]
    }


@pytest.fixture
def mock_rekognition_text_response():
    """Create a mock Rekognition text detection response."""
    return {
        'TextDetections': [
            {
                'DetectedText': 'HELLO',
                'Type': 'LINE',
                'Confidence': 99.1
            },
            {
                'DetectedText': 'WORLD',
                'Type': 'LINE',
                'Confidence': 98.5
            }
        ]
    }


class TestImageProcessingService:
    """Tests for ImageProcessingService."""
    
    def test_init(self, image_processing_service):
        """Test service initialization."""
        assert isinstance(image_processing_service, ImageProcessingService)
    
    @patch("app.aws.image_processing.get_textract_client")
    def test_extract_text_from_image_sync(
        self,
        mock_get_textract_client,
        mock_textract_response,
        image_processing_service
    ):
        """Test synchronous text extraction."""
        # Set up mock
        mock_client = MagicMock()
        mock_client.detect_document_text.return_value = mock_textract_response
        mock_get_textract_client.return_value = mock_client
        
        # Extract text
        result = image_processing_service.extract_text_from_image_sync(b"test image")
        
        # Verify result
        assert result == "This is a test\nOCR extraction"
        
        # Verify mock was called correctly
        mock_client.detect_document_text.assert_called_once_with(
            Document={'Bytes': b"test image"}
        )
    
    @patch("app.aws.image_processing.get_textract_client")
    def test_extract_text_from_image_sync_error(
        self,
        mock_get_textract_client,
        image_processing_service
    ):
        """Test synchronous text extraction with error."""
        # Set up mock to raise error
        mock_client = MagicMock()
        mock_error = ClientError(
            {'Error': {'Code': 'InvalidImageFormatException', 'Message': 'Invalid image'}},
            'DetectDocumentText'
        )
        mock_client.detect_document_text.side_effect = mock_error
        mock_get_textract_client.return_value = mock_client
        
        # Extract text with error
        with pytest.raises(TextractError):
            image_processing_service.extract_text_from_image_sync(b"test image")
    
    @pytest.mark.asyncio
    @patch("app.aws.image_processing.get_async_textract_client")
    async def test_extract_text_from_image(
        self,
        mock_get_async_textract_client,
        mock_textract_response,
        image_processing_service
    ):
        """Test asynchronous text extraction."""
        # Set up mock
        mock_client = AsyncMock()
        mock_client.detect_document_text.return_value = mock_textract_response
        
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_client
        mock_client_context.__aexit__.return_value = None
        
        mock_get_async_textract_client.return_value = lambda: mock_client_context
        
        # Extract text
        result = await image_processing_service.extract_text_from_image(b"test image")
        
        # Verify result
        assert result == "This is a test\nOCR extraction"
        
        # Verify mock was called correctly
        mock_client.detect_document_text.assert_called_once_with(
            Document={'Bytes': b"test image"}
        )
    
    @patch("app.aws.image_processing.get_rekognition_client")
    def test_describe_image_sync(
        self,
        mock_get_rekognition_client,
        mock_rekognition_label_response,
        mock_rekognition_text_response,
        image_processing_service
    ):
        """Test synchronous image description."""
        # Set up mock
        mock_client = MagicMock()
        mock_client.detect_labels.return_value = mock_rekognition_label_response
        mock_client.detect_text.return_value = mock_rekognition_text_response
        mock_get_rekognition_client.return_value = mock_client
        
        # Describe image
        result = image_processing_service.describe_image_sync(b"test image")
        
        # Verify result
        assert "labels" in result
        assert "detected_text" in result
        assert "description" in result
        assert len(result["labels"]) == 3
        assert result["detected_text"] == "HELLO WORLD"
        assert "Person" in result["description"]
        assert "Mountain" in result["description"]
        assert "Tree" in result["description"]
        
        # Verify mocks were called correctly
        mock_client.detect_labels.assert_called_once()
        mock_client.detect_text.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("app.aws.image_processing.get_async_rekognition_client")
    async def test_describe_image(
        self,
        mock_get_async_rekognition_client,
        mock_rekognition_label_response,
        mock_rekognition_text_response,
        image_processing_service
    ):
        """Test asynchronous image description."""
        # Set up mock
        mock_client = AsyncMock()
        mock_client.detect_labels.return_value = mock_rekognition_label_response
        mock_client.detect_text.return_value = mock_rekognition_text_response
        
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_client
        mock_client_context.__aexit__.return_value = None
        
        mock_get_async_rekognition_client.return_value = lambda: mock_client_context
        
        # Describe image
        result = await image_processing_service.describe_image(b"test image")
        
        # Verify result
        assert "labels" in result
        assert "detected_text" in result
        assert "description" in result
        assert len(result["labels"]) == 3
        assert result["detected_text"] == "HELLO WORLD"
        assert "Person" in result["description"]
        assert "Mountain" in result["description"]
        assert "Tree" in result["description"]
        
        # Verify mocks were called correctly
        mock_client.detect_labels.assert_called_once()
        mock_client.detect_text.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("app.aws.image_processing.ImageProcessingService.extract_text_from_image")
    @patch("app.aws.image_processing.ImageProcessingService.describe_image")
    async def test_process_image_success(
        self,
        mock_describe_image,
        mock_extract_text,
        mock_image_file,
        image_processing_service
    ):
        """Test image processing with both services succeeding."""
        # Set up mocks
        mock_extract_text.return_value = "This is extracted text from the image."
        mock_describe_image.return_value = {
            'labels': [
                {'name': 'Person', 'confidence': 99.2, 'parents': []},
                {'name': 'Mountain', 'confidence': 95.8, 'parents': [{'name': 'Nature'}]}
            ],
            'detected_text': 'HELLO WORLD',
            'description': 'Image contains: Person, Mountain'
        }
        
        # Process image
        file_id = uuid.uuid4()
        result = await image_processing_service.process_image(mock_image_file, file_id)
        
        # Verify result
        assert isinstance(result, ProcessedPhoto)
        assert result.original_id == file_id
        assert result.processed_text == "This is extracted text from the image."
        assert result.content_type == "image_text"
        assert result.processing_status == "completed"
        assert result.extracted_text == "This is extracted text from the image."
        assert result.image_description == "Image contains: Person, Mountain"
        assert result.detected_objects == ["Person", "Mountain"]
        assert result.error_details is None
        
        # Verify mocks were called correctly
        mock_extract_text.assert_called_once()
        mock_describe_image.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("app.aws.image_processing.ImageProcessingService.extract_text_from_image")
    @patch("app.aws.image_processing.ImageProcessingService.describe_image")
    async def test_process_image_textract_failure(
        self,
        mock_describe_image,
        mock_extract_text,
        mock_image_file,
        image_processing_service
    ):
        """Test image processing with Textract failing but Rekognition succeeding."""
        # Set up mocks
        mock_extract_text.side_effect = TextractError(
            message="Textract processing failed",
            operation="extract_text_from_image"
        )
        mock_describe_image.return_value = {
            'labels': [
                {'name': 'Person', 'confidence': 99.2, 'parents': []},
                {'name': 'Mountain', 'confidence': 95.8, 'parents': [{'name': 'Nature'}]}
            ],
            'detected_text': 'HELLO WORLD',
            'description': 'Image contains: Person, Mountain'
        }
        
        # Process image
        result = await image_processing_service.process_image(mock_image_file)
        
        # Verify result
        assert isinstance(result, ProcessedPhoto)
        assert result.processing_status == "partial"
        assert result.content_type == "image_desc"
        assert result.extracted_text == "HELLO WORLD"
        assert result.processed_text == "Image contains: Person, Mountain\nHELLO WORLD"
        assert result.image_description == "Image contains: Person, Mountain"
        assert result.detected_objects == ["Person", "Mountain"]
        
        # Verify mocks were called correctly
        mock_extract_text.assert_called_once()
        mock_describe_image.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("app.aws.image_processing.ImageProcessingService.extract_text_from_image")
    @patch("app.aws.image_processing.ImageProcessingService.describe_image")
    async def test_process_image_rekognition_failure(
        self,
        mock_describe_image,
        mock_extract_text,
        mock_image_file,
        image_processing_service
    ):
        """Test image processing with Rekognition failing but Textract succeeding."""
        # Set up mocks
        mock_extract_text.return_value = "This is extracted text from the image."
        mock_describe_image.side_effect = RekognitionError(
            message="Rekognition processing failed",
            operation="describe_image"
        )
        
        # Process image
        result = await image_processing_service.process_image(mock_image_file)
        
        # Verify result
        assert isinstance(result, ProcessedPhoto)
        assert result.processing_status == "partial"
        assert result.content_type == "image_text"
        assert result.extracted_text == "This is extracted text from the image."
        assert result.processed_text == "This is extracted text from the image."
        assert result.image_description is None
        assert result.detected_objects is None
        
        # Verify mocks were called correctly
        mock_extract_text.assert_called_once()
        mock_describe_image.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("app.aws.image_processing.ImageProcessingService.extract_text_from_image")
    @patch("app.aws.image_processing.ImageProcessingService.describe_image")
    async def test_process_image_both_services_failure(
        self,
        mock_describe_image,
        mock_extract_text,
        mock_image_file,
        image_processing_service
    ):
        """Test image processing with both services failing."""
        # Set up mocks
        mock_extract_text.side_effect = TextractError(
            message="Textract processing failed",
            operation="extract_text_from_image"
        )
        mock_describe_image.side_effect = RekognitionError(
            message="Rekognition processing failed",
            operation="describe_image"
        )
        
        # Process image
        result = await image_processing_service.process_image(mock_image_file)
        
        # Verify result
        assert isinstance(result, ProcessedPhoto)
        assert result.processing_status == "failed"
        assert result.processed_text == "Image processing failed"
        assert result.extracted_text is None
        assert result.image_description is None
        assert result.detected_objects is None
        assert "Textract processing failed" in result.error_details
        assert "Rekognition processing failed" in result.error_details
        
        # Verify mocks were called correctly
        mock_extract_text.assert_called_once()
        mock_describe_image.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("app.aws.image_processing.ImageProcessingService.extract_text_from_image")
    @patch("app.aws.image_processing.ImageProcessingService.describe_image")
    async def test_process_image_no_content(
        self,
        mock_describe_image,
        mock_extract_text,
        mock_image_file,
        image_processing_service
    ):
        """Test image processing with no meaningful content extracted."""
        # Set up mocks
        mock_extract_text.return_value = ""
        mock_describe_image.return_value = {
            'labels': [],
            'detected_text': '',
            'description': 'Image contains: '
        }
        
        # Process image
        result = await image_processing_service.process_image(mock_image_file)
        
        # Verify result
        assert isinstance(result, ProcessedPhoto)
        assert result.processing_status == "partial"
        assert result.processed_text == "No text or recognizable content found in image"
        
        # Verify mocks were called correctly
        mock_extract_text.assert_called_once()
        mock_describe_image.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_image_with_bytes(
        self,
        image_processing_service
    ):
        """Test processing an image provided as bytes."""
        # Create test image bytes
        image_bytes = b"test image bytes"
        
        # Mock the extract_text_from_image and describe_image methods
        image_processing_service.extract_text_from_image = AsyncMock(return_value="Extracted text")
        image_processing_service.describe_image = AsyncMock(return_value={
            'labels': [{'name': 'Test', 'confidence': 99.0, 'parents': []}],
            'detected_text': 'Test text',
            'description': 'Image contains: Test'
        })
        
        # Process image
        result = await image_processing_service.process_image(image_bytes)
        
        # Verify result
        assert isinstance(result, ProcessedPhoto)
        assert result.processing_status == "completed"
        assert result.processed_text == "Extracted text"
        assert result.extracted_text == "Extracted text"
        assert result.image_description == "Image contains: Test"
        assert result.detected_objects == ["Test"]
        
        # Verify mocks were called correctly
        image_processing_service.extract_text_from_image.assert_called_once_with(image_bytes)
        image_processing_service.describe_image.assert_called_once_with(image_bytes)