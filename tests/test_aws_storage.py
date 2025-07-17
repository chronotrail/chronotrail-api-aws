"""
Tests for S3 file storage service.
"""
import os
import io
import uuid
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi import UploadFile, HTTPException
from botocore.exceptions import ClientError

from app.aws.storage import FileStorageService
from app.aws.exceptions import S3Error


@pytest.fixture
def mock_upload_file():
    """Create a mock UploadFile for testing."""
    file_content = b"test file content"
    file = MagicMock(spec=UploadFile)
    file.filename = "test.jpg"
    file.content_type = "image/jpeg"
    file.file = io.BytesIO(file_content)
    file.read = AsyncMock(return_value=file_content)
    file.seek = AsyncMock()
    file.tell = AsyncMock(return_value=len(file_content))
    return file


@pytest.fixture
def storage_service():
    """Create a FileStorageService instance for testing."""
    return FileStorageService(bucket_name="test-bucket")


@pytest.mark.asyncio
class TestFileStorageService:
    """Tests for FileStorageService."""
    
    @patch("app.aws.storage.validate_file")
    @patch("app.aws.storage.upload_file_to_s3")
    @patch("app.aws.storage.generate_s3_key")
    async def test_upload_file(
        self, 
        mock_generate_s3_key, 
        mock_upload_file_to_s3, 
        mock_validate_file,
        mock_upload_file,
        storage_service
    ):
        """Test file upload."""
        # Set up mocks
        mock_validate_file.return_value = (True, None)
        mock_generate_s3_key.return_value = "user-123/photo/2023/01/01/test-uuid.jpg"
        mock_upload_file_to_s3.return_value = "https://test-bucket.s3.amazonaws.com/user-123/photo/2023/01/01/test-uuid.jpg"
        
        # Upload file
        user_id = "user-123"
        file_type = "photo"
        timestamp = datetime(2023, 1, 1)
        
        result = await storage_service.upload_file(
            file=mock_upload_file,
            user_id=user_id,
            file_type=file_type,
            timestamp=timestamp
        )
        
        # Verify result
        assert result["bucket_name"] == "test-bucket"
        assert result["s3_key"] == "user-123/photo/2023/01/01/test-uuid.jpg"
        assert result["s3_url"] == "https://test-bucket.s3.amazonaws.com/user-123/photo/2023/01/01/test-uuid.jpg"
        assert result["file_type"] == "photo"
        assert result["original_filename"] == "test.jpg"
        assert result["content_type"] == "image/jpeg"
        assert result["timestamp"] == timestamp
        
        # Verify mocks were called correctly
        mock_validate_file.assert_called_once()
        mock_generate_s3_key.assert_called_once_with(
            user_id=user_id,
            file_type=file_type,
            original_filename="test.jpg",
            timestamp=timestamp
        )
        mock_upload_file_to_s3.assert_called_once()
    
    @patch("app.aws.storage.validate_file")
    async def test_upload_file_validation_failure(
        self,
        mock_validate_file,
        mock_upload_file,
        storage_service
    ):
        """Test file upload with validation failure."""
        # Set up mock to fail validation
        mock_validate_file.return_value = (False, "Invalid file type")
        
        # Attempt to upload file
        with pytest.raises(HTTPException) as excinfo:
            await storage_service.upload_file(
                file=mock_upload_file,
                user_id="user-123",
                file_type="photo"
            )
        
        # Verify exception
        assert excinfo.value.status_code == 400
        assert "Invalid file type" in str(excinfo.value.detail)
    
    @patch("app.aws.storage.check_file_exists")
    @patch("app.aws.storage.download_file_from_s3")
    async def test_download_file(
        self,
        mock_download_file_from_s3,
        mock_check_file_exists,
        storage_service
    ):
        """Test file download."""
        # Set up mocks
        mock_check_file_exists.return_value = True
        mock_download_file_from_s3.return_value = b"test file content"
        
        # Download file
        result = await storage_service.download_file("test-key")
        
        # Verify result
        assert result == b"test file content"
        
        # Verify mocks were called correctly
        mock_check_file_exists.assert_called_once_with("test-bucket", "test-key")
        mock_download_file_from_s3.assert_called_once_with(
            bucket_name="test-bucket",
            object_key="test-key"
        )
    
    @patch("app.aws.storage.check_file_exists")
    async def test_download_file_not_found(
        self,
        mock_check_file_exists,
        storage_service
    ):
        """Test file download when file doesn't exist."""
        # Set up mock to indicate file doesn't exist
        mock_check_file_exists.return_value = False
        
        # Attempt to download file
        with pytest.raises(HTTPException) as excinfo:
            await storage_service.download_file("test-key")
        
        # Verify exception
        assert excinfo.value.status_code == 404
        assert "File not found" in str(excinfo.value.detail)
    
    @patch("app.aws.storage.delete_file_from_s3")
    async def test_delete_file(
        self,
        mock_delete_file_from_s3,
        storage_service
    ):
        """Test file deletion."""
        # Set up mock
        mock_delete_file_from_s3.return_value = True
        
        # Delete file
        result = await storage_service.delete_file("test-key")
        
        # Verify result
        assert result is True
        
        # Verify mock was called correctly
        mock_delete_file_from_s3.assert_called_once_with(
            bucket_name="test-bucket",
            object_key="test-key"
        )
    
    @patch("app.aws.storage.check_file_exists")
    @patch("app.aws.storage.get_s3_client")
    def test_generate_download_url(
        self,
        mock_get_s3_client,
        mock_check_file_exists,
        storage_service
    ):
        """Test download URL generation."""
        # Set up mocks
        mock_check_file_exists.return_value = True
        mock_s3_client = MagicMock()
        mock_s3_client.generate_presigned_url.return_value = "https://presigned-url.com"
        mock_get_s3_client.return_value = mock_s3_client
        
        # Generate URL
        result = storage_service.generate_download_url(
            s3_key="test-key",
            expiration=1800,
            file_name="download.jpg"
        )
        
        # Verify result
        assert result == "https://presigned-url.com"
        
        # Verify mocks were called correctly
        mock_check_file_exists.assert_called_once_with("test-bucket", "test-key")
        mock_s3_client.generate_presigned_url.assert_called_once_with(
            ClientMethod='get_object',
            Params={
                'Bucket': 'test-bucket',
                'Key': 'test-key',
                'ResponseContentDisposition': 'attachment; filename="download.jpg"'
            },
            ExpiresIn=1800,
            HttpMethod='GET'
        )
    
    @patch("app.aws.storage.check_file_exists")
    def test_generate_download_url_file_not_found(
        self,
        mock_check_file_exists,
        storage_service
    ):
        """Test download URL generation when file doesn't exist."""
        # Set up mock to indicate file doesn't exist
        mock_check_file_exists.return_value = False
        
        # Attempt to generate URL
        with pytest.raises(HTTPException) as excinfo:
            storage_service.generate_download_url("test-key")
        
        # Verify exception
        assert excinfo.value.status_code == 404
        assert "File not found" in str(excinfo.value.detail)
    
    @patch("app.aws.storage.get_s3_client")
    def test_generate_upload_url(
        self,
        mock_get_s3_client,
        storage_service
    ):
        """Test upload URL generation."""
        # Set up mock
        mock_s3_client = MagicMock()
        mock_s3_client.generate_presigned_post.return_value = {
            'url': 'https://test-bucket.s3.amazonaws.com',
            'fields': {
                'key': 'test-key',
                'Content-Type': 'image/jpeg',
                'policy': 'policy-data',
                'x-amz-signature': 'signature-data'
            }
        }
        mock_get_s3_client.return_value = mock_s3_client
        
        # Generate URL
        result = storage_service.generate_upload_url(
            s3_key="test-key",
            content_type="image/jpeg",
            expiration=1800,
            max_size=5 * 1024 * 1024
        )
        
        # Verify result
        assert result['url'] == 'https://test-bucket.s3.amazonaws.com'
        assert 'fields' in result
        assert result['fields']['key'] == 'test-key'
        
        # Verify mock was called correctly
        mock_s3_client.generate_presigned_post.assert_called_once()
    
    @patch("app.aws.storage.check_file_exists")
    def test_check_file_exists(
        self,
        mock_check_file_exists,
        storage_service
    ):
        """Test file existence check."""
        # Set up mock
        mock_check_file_exists.return_value = True
        
        # Check if file exists
        result = storage_service.check_file_exists("test-key")
        
        # Verify result
        assert result is True
        
        # Verify mock was called correctly
        mock_check_file_exists.assert_called_once_with("test-bucket", "test-key")
    
    @patch("app.aws.storage.check_file_exists")
    @patch("app.aws.storage.get_s3_client")
    def test_get_file_metadata(
        self,
        mock_get_s3_client,
        mock_check_file_exists,
        storage_service
    ):
        """Test file metadata retrieval."""
        # Set up mocks
        mock_check_file_exists.return_value = True
        mock_s3_client = MagicMock()
        mock_s3_client.head_object.return_value = {
            'Metadata': {'user_id': 'user-123'},
            'ContentType': 'image/jpeg',
            'ContentLength': 1024,
            'LastModified': datetime(2023, 1, 1)
        }
        mock_get_s3_client.return_value = mock_s3_client
        
        # Get metadata
        result = storage_service.get_file_metadata("test-key")
        
        # Verify result
        assert result['metadata'] == {'user_id': 'user-123'}
        assert result['content_type'] == 'image/jpeg'
        assert result['content_length'] == 1024
        assert result['last_modified'] == '2023-01-01T00:00:00'
        
        # Verify mocks were called correctly
        mock_check_file_exists.assert_called_once_with("test-bucket", "test-key")
        mock_s3_client.head_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test-key"
        )
    
    @patch("app.aws.storage.check_file_exists")
    def test_get_file_metadata_file_not_found(
        self,
        mock_check_file_exists,
        storage_service
    ):
        """Test file metadata retrieval when file doesn't exist."""
        # Set up mock to indicate file doesn't exist
        mock_check_file_exists.return_value = False
        
        # Attempt to get metadata
        with pytest.raises(HTTPException) as excinfo:
            storage_service.get_file_metadata("test-key")
        
        # Verify exception
        assert excinfo.value.status_code == 404
        assert "File not found" in str(excinfo.value.detail)