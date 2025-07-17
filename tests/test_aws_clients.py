"""
Tests for AWS client configuration and utilities.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

import boto3
import botocore.session
from botocore.stub import Stubber
from botocore.exceptions import ClientError

from app.aws.clients import (
    aws_client_manager,
    get_s3_client,
    get_textract_client,
    get_rekognition_client,
    get_transcribe_client,
    get_bedrock_client,
    handle_aws_error,
)
from app.aws.utils import (
    generate_s3_key,
    generate_presigned_url,
    is_valid_s3_url,
    extract_bucket_and_key_from_url,
    check_file_exists,
)


class TestAWSClients(unittest.TestCase):
    """Test AWS client configuration and utilities."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear cached clients
        aws_client_manager._clients = {}
        aws_client_manager._session = None
        
        # Set test environment variables
        os.environ["AWS_REGION"] = "us-east-1"
        os.environ["AWS_ACCESS_KEY_ID"] = "test-key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test-secret"
        
    def tearDown(self):
        """Clean up test environment."""
        # Clear cached clients
        aws_client_manager._clients = {}
        aws_client_manager._session = None
    
    def test_client_creation(self):
        """Test that clients are created correctly."""
        # Test S3 client
        s3_client = get_s3_client()
        self.assertIsNotNone(s3_client)
        self.assertEqual(s3_client.meta.service_model.service_name, "s3")
        
        # Test Textract client
        textract_client = get_textract_client()
        self.assertIsNotNone(textract_client)
        self.assertEqual(textract_client.meta.service_model.service_name, "textract")
        
        # Test Rekognition client
        rekognition_client = get_rekognition_client()
        self.assertIsNotNone(rekognition_client)
        self.assertEqual(rekognition_client.meta.service_model.service_name, "rekognition")
        
        # Test Transcribe client
        transcribe_client = get_transcribe_client()
        self.assertIsNotNone(transcribe_client)
        self.assertEqual(transcribe_client.meta.service_model.service_name, "transcribe")
        
        # Test Bedrock client
        bedrock_client = get_bedrock_client()
        self.assertIsNotNone(bedrock_client)
        self.assertEqual(bedrock_client.meta.service_model.service_name, "bedrock-runtime")
    
    def test_client_caching(self):
        """Test that clients are cached."""
        # Get clients twice
        s3_client1 = get_s3_client()
        s3_client2 = get_s3_client()
        
        # Verify they are the same instance
        self.assertIs(s3_client1, s3_client2)
    
    def test_error_handling_decorator(self):
        """Test the error handling decorator."""
        # Create a mock function that raises a ClientError
        mock_response = {
            "Error": {
                "Code": "TestError",
                "Message": "Test error message"
            }
        }
        mock_error = ClientError(
            mock_response, "test_operation"
        )
        
        @handle_aws_error()
        def test_func():
            raise mock_error
        
        # Verify the decorator properly re-raises the exception
        with self.assertRaises(ClientError):
            test_func()
    
    def test_generate_s3_key(self):
        """Test S3 key generation."""
        user_id = "test-user"
        file_type = "photo"
        original_filename = "test.jpg"
        
        # Generate key
        key = generate_s3_key(user_id, file_type, original_filename)
        
        # Verify key format
        self.assertTrue(key.startswith(f"{user_id}/{file_type}/"))
        self.assertTrue(key.endswith(".jpg"))
    
    @patch("app.aws.utils.get_s3_client")
    def test_generate_presigned_url(self, mock_get_s3_client):
        """Test presigned URL generation."""
        # Set up mock
        mock_client = MagicMock()
        mock_client.generate_presigned_url.return_value = "https://test-url.com"
        mock_get_s3_client.return_value = mock_client
        
        # Generate URL
        url = generate_presigned_url("test-bucket", "test-key")
        
        # Verify URL
        self.assertEqual(url, "https://test-url.com")
        mock_client.generate_presigned_url.assert_called_once()
    
    def test_is_valid_s3_url(self):
        """Test S3 URL validation."""
        # Valid URLs
        self.assertTrue(is_valid_s3_url("https://bucket.s3.amazonaws.com/key"))
        self.assertTrue(is_valid_s3_url("https://s3.amazonaws.com/bucket/key"))
        self.assertTrue(is_valid_s3_url("https://bucket.s3.us-east-1.amazonaws.com/key"))
        
        # Invalid URLs
        self.assertFalse(is_valid_s3_url("https://example.com/key"))
        self.assertFalse(is_valid_s3_url("not-a-url"))
    
    def test_extract_bucket_and_key(self):
        """Test bucket and key extraction from S3 URLs."""
        # Test different URL formats
        bucket, key = extract_bucket_and_key_from_url("https://bucket.s3.amazonaws.com/key")
        self.assertEqual(bucket, "bucket")
        self.assertEqual(key, "key")
        
        bucket, key = extract_bucket_and_key_from_url("https://s3.amazonaws.com/bucket/key")
        self.assertEqual(bucket, "bucket")
        self.assertEqual(key, "key")
        
        bucket, key = extract_bucket_and_key_from_url("https://bucket.s3.us-east-1.amazonaws.com/key")
        self.assertEqual(bucket, "bucket")
        self.assertEqual(key, "key")
        
        # Test with invalid URL
        with self.assertRaises(ValueError):
            extract_bucket_and_key_from_url("https://example.com/key")
    
    @patch("app.aws.utils.get_s3_client")
    def test_check_file_exists(self, mock_get_s3_client):
        """Test file existence check."""
        # Set up mock for existing file
        mock_client = MagicMock()
        mock_client.head_object.return_value = {}
        mock_get_s3_client.return_value = mock_client
        
        # Check existing file
        exists = check_file_exists("test-bucket", "test-key")
        self.assertTrue(exists)
        
        # Set up mock for non-existing file
        error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
        mock_client.head_object.side_effect = ClientError(
            error_response, "head_object"
        )
        
        # Check non-existing file
        exists = check_file_exists("test-bucket", "test-key")
        self.assertFalse(exists)


if __name__ == "__main__":
    unittest.main()