"""
Tests for speech-to-text transcription service.
"""

import io
import json
import unittest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from fastapi import UploadFile

from app.aws.exceptions import FileProcessingError, S3Error, TranscribeError
from app.aws.transcription import TranscriptionService
from app.models.schemas import ProcessedVoice


@pytest.fixture
def transcription_service():
    """Create a TranscriptionService instance for testing."""
    return TranscriptionService(bucket_name="test-bucket")


@pytest.fixture
def mock_audio_file():
    """Create a mock audio file for testing."""
    file_content = b"test audio content"
    file = MagicMock(spec=UploadFile)
    file.filename = "test.mp3"
    file.content_type = "audio/mpeg"
    file.file = io.BytesIO(file_content)
    file.read = AsyncMock(return_value=file_content)
    file.seek = AsyncMock()
    return file


@pytest.fixture
def mock_transcribe_response():
    """Create a mock Transcribe response."""
    return {
        "TranscriptionJobName": "test-job",
        "TranscriptionJobStatus": "COMPLETED",
        "LanguageCode": "en-US",
        "MediaFormat": "mp3",
        "Media": {"MediaFileUri": "s3://test-bucket/transcribe/test-job.mp3"},
        "Transcript": {
            "TranscriptFileUri": "s3://test-bucket/transcribe/test-job.json"
        },
    }


@pytest.fixture
def mock_transcript_data():
    """Create mock transcript data."""
    return {
        "jobName": "test-job",
        "results": {
            "transcripts": [{"transcript": "This is a test transcription."}],
            "items": [
                {
                    "start_time": "0.0",
                    "end_time": "1.0",
                    "alternatives": [{"confidence": "0.95", "content": "This"}],
                    "type": "pronunciation",
                },
                {
                    "start_time": "1.1",
                    "end_time": "1.5",
                    "alternatives": [{"confidence": "0.98", "content": "is"}],
                    "type": "pronunciation",
                },
                {
                    "start_time": "1.6",
                    "end_time": "1.8",
                    "alternatives": [{"confidence": "0.97", "content": "a"}],
                    "type": "pronunciation",
                },
                {
                    "start_time": "1.9",
                    "end_time": "2.3",
                    "alternatives": [{"confidence": "0.99", "content": "test"}],
                    "type": "pronunciation",
                },
                {
                    "start_time": "2.4",
                    "end_time": "3.5",
                    "alternatives": [
                        {"confidence": "0.96", "content": "transcription"}
                    ],
                    "type": "pronunciation",
                },
                {
                    "alternatives": [{"confidence": "0.0", "content": "."}],
                    "type": "punctuation",
                },
            ],
        },
        "status": "COMPLETED",
    }


class TestTranscriptionService:
    """Tests for TranscriptionService."""

    def test_init(self, transcription_service):
        """Test service initialization."""
        assert isinstance(transcription_service, TranscriptionService)
        assert transcription_service.bucket_name == "test-bucket"

    @patch("app.aws.transcription.get_transcribe_client")
    def test_start_transcription_job_sync(
        self, mock_get_transcribe_client, transcription_service
    ):
        """Test synchronous transcription job start."""
        # Set up mock
        mock_client = MagicMock()
        mock_client.start_transcription_job.return_value = {
            "TranscriptionJob": {
                "TranscriptionJobName": "test-job",
                "TranscriptionJobStatus": "IN_PROGRESS",
            }
        }
        mock_get_transcribe_client.return_value = mock_client

        # Start job
        result = transcription_service.start_transcription_job_sync(
            job_name="test-job",
            media_uri="s3://test-bucket/test.mp3",
            media_format="mp3",
        )

        # Verify result
        assert result["TranscriptionJobName"] == "test-job"
        assert result["TranscriptionJobStatus"] == "IN_PROGRESS"

        # Verify mock was called correctly
        mock_client.start_transcription_job.assert_called_once()

    @patch("app.aws.transcription.get_transcribe_client")
    def test_get_transcription_job_sync(
        self,
        mock_get_transcribe_client,
        mock_transcribe_response,
        transcription_service,
    ):
        """Test synchronous transcription job retrieval."""
        # Set up mock
        mock_client = MagicMock()
        mock_client.get_transcription_job.return_value = {
            "TranscriptionJob": mock_transcribe_response
        }
        mock_get_transcribe_client.return_value = mock_client

        # Get job
        result = transcription_service.get_transcription_job_sync("test-job")

        # Verify result
        assert result == mock_transcribe_response

        # Verify mock was called correctly
        mock_client.get_transcription_job.assert_called_once_with(
            TranscriptionJobName="test-job"
        )

    @pytest.mark.asyncio
    @patch("app.aws.transcription.get_async_transcribe_client")
    async def test_start_transcription_job(
        self, mock_get_async_transcribe_client, transcription_service
    ):
        """Test asynchronous transcription job start."""
        # Set up mock
        mock_client = AsyncMock()
        mock_client.start_transcription_job.return_value = {
            "TranscriptionJob": {
                "TranscriptionJobName": "test-job",
                "TranscriptionJobStatus": "IN_PROGRESS",
            }
        }

        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_client
        mock_client_context.__aexit__.return_value = None

        mock_get_async_transcribe_client.return_value = lambda: mock_client_context

        # Start job
        result = await transcription_service.start_transcription_job(
            job_name="test-job",
            media_uri="s3://test-bucket/test.mp3",
            media_format="mp3",
        )

        # Verify result
        assert result["TranscriptionJobName"] == "test-job"
        assert result["TranscriptionJobStatus"] == "IN_PROGRESS"

        # Verify mock was called correctly
        mock_client.start_transcription_job.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.aws.transcription.get_async_transcribe_client")
    async def test_get_transcription_job(
        self,
        mock_get_async_transcribe_client,
        mock_transcribe_response,
        transcription_service,
    ):
        """Test asynchronous transcription job retrieval."""
        # Set up mock
        mock_client = AsyncMock()
        mock_client.get_transcription_job.return_value = {
            "TranscriptionJob": mock_transcribe_response
        }

        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_client
        mock_client_context.__aexit__.return_value = None

        mock_get_async_transcribe_client.return_value = lambda: mock_client_context

        # Get job
        result = await transcription_service.get_transcription_job("test-job")

        # Verify result
        assert result == mock_transcribe_response

        # Verify mock was called correctly
        mock_client.get_transcription_job.assert_called_once_with(
            TranscriptionJobName="test-job"
        )

    @pytest.mark.asyncio
    @patch("app.aws.transcription.TranscriptionService.get_transcription_job")
    @patch("app.aws.transcription.asyncio.sleep")
    async def test_wait_for_transcription_job_completed(
        self,
        mock_sleep,
        mock_get_transcription_job,
        mock_transcribe_response,
        transcription_service,
    ):
        """Test waiting for a transcription job that completes successfully."""
        # Set up mock
        mock_get_transcription_job.return_value = mock_transcribe_response

        # Wait for job
        result = await transcription_service.wait_for_transcription_job("test-job")

        # Verify result
        assert result == mock_transcribe_response

        # Verify mocks were called correctly
        mock_get_transcription_job.assert_called_once_with("test-job")
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.aws.transcription.TranscriptionService.get_transcription_job")
    @patch("app.aws.transcription.asyncio.sleep")
    async def test_wait_for_transcription_job_failed(
        self, mock_sleep, mock_get_transcription_job, transcription_service
    ):
        """Test waiting for a transcription job that fails."""
        # Set up mock for failed job
        mock_get_transcription_job.return_value = {
            "TranscriptionJobStatus": "FAILED",
            "FailureReason": "Test failure reason",
        }

        # Wait for job with expected failure
        with pytest.raises(TranscribeError) as excinfo:
            await transcription_service.wait_for_transcription_job("test-job")

        # Verify exception
        assert "Test failure reason" in str(excinfo.value)

        # Verify mocks were called correctly
        mock_get_transcription_job.assert_called_once_with("test-job")
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.aws.transcription.TranscriptionService.get_transcription_job")
    @patch("app.aws.transcription.asyncio.sleep")
    @patch("app.aws.transcription.time.time")
    async def test_wait_for_transcription_job_timeout(
        self, mock_time, mock_sleep, mock_get_transcription_job, transcription_service
    ):
        """Test waiting for a transcription job that times out."""
        # Set up mocks
        mock_time.side_effect = [
            0,
            301,
        ]  # First call returns 0, second call returns 301 (> 300 timeout)
        mock_get_transcription_job.return_value = {
            "TranscriptionJobStatus": "IN_PROGRESS"
        }

        # Wait for job with expected timeout
        with pytest.raises(TranscribeError) as excinfo:
            await transcription_service.wait_for_transcription_job(
                "test-job", timeout=300
            )

        # Verify exception
        assert "timed out" in str(excinfo.value)

        # Verify mocks were called correctly
        mock_get_transcription_job.assert_not_called()  # Should timeout before first call

    @pytest.mark.asyncio
    @patch("app.aws.transcription.get_async_s3_client")
    async def test_download_transcript(
        self, mock_get_async_s3_client, mock_transcript_data, transcription_service
    ):
        """Test transcript download and parsing."""
        # Set up mock
        mock_client = AsyncMock()

        # Mock download_fileobj to write data to the file object
        async def mock_download_fileobj(bucket, key, file_obj):
            file_obj.write(json.dumps(mock_transcript_data).encode("utf-8"))

        mock_client.download_fileobj.side_effect = mock_download_fileobj

        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_client
        mock_client_context.__aexit__.return_value = None

        mock_get_async_s3_client.return_value = lambda: mock_client_context

        # Download transcript
        result = await transcription_service.download_transcript(
            "s3://test-bucket/transcribe/test-job.json"
        )

        # Verify result
        assert result == mock_transcript_data

        # Verify mock was called correctly
        mock_client.download_fileobj.assert_called_once_with(
            "test-bucket", "transcribe/test-job.json", unittest.mock.ANY
        )

    def test_extract_transcript_text(self, mock_transcript_data, transcription_service):
        """Test transcript text extraction."""
        # Extract text
        text, confidence = transcription_service.extract_transcript_text(
            mock_transcript_data
        )

        # Verify results
        assert text == "This is a test transcription."
        assert confidence == pytest.approx(
            0.97, abs=0.01
        )  # Average of confidence scores

    def test_extract_transcript_text_empty(self, transcription_service):
        """Test transcript text extraction with empty data."""
        # Extract text from empty data
        text, confidence = transcription_service.extract_transcript_text({})

        # Verify results
        assert text == ""
        assert confidence == 0.0

    @pytest.mark.asyncio
    @patch("app.aws.transcription.TranscriptionService.start_transcription_job")
    @patch("app.aws.transcription.TranscriptionService.wait_for_transcription_job")
    @patch("app.aws.transcription.TranscriptionService.download_transcript")
    @patch("app.aws.transcription.TranscriptionService.extract_transcript_text")
    @patch("app.aws.transcription.get_async_s3_client")
    @patch("uuid.uuid4")
    async def test_transcribe_audio_file_with_upload_file(
        self,
        mock_uuid4,
        mock_get_async_s3_client,
        mock_extract_transcript_text,
        mock_download_transcript,
        mock_wait_for_transcription_job,
        mock_start_transcription_job,
        mock_audio_file,
        mock_transcribe_response,
        mock_transcript_data,
        transcription_service,
    ):
        """Test complete audio file transcription workflow with UploadFile."""
        # Set up mocks
        mock_uuid4.return_value = "test-uuid"

        mock_client = AsyncMock()
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_client
        mock_client_context.__aexit__.return_value = None
        mock_get_async_s3_client.return_value = lambda: mock_client_context

        mock_start_transcription_job.return_value = {
            "TranscriptionJobName": "test-job",
            "TranscriptionJobStatus": "IN_PROGRESS",
        }
        # Set up the mock to return the response directly
        mock_wait_for_transcription_job.return_value = mock_transcribe_response
        mock_download_transcript.return_value = mock_transcript_data
        mock_extract_transcript_text.return_value = (
            "This is a test transcription.",
            0.97,
        )

        # Transcribe audio file
        result = await transcription_service.transcribe_audio_file(
            file=mock_audio_file, job_name="test-job"
        )

        # Verify result
        assert result["transcript"] == "This is a test transcription."
        assert result["confidence"] == 0.97
        assert result["job_name"] == "test-job"
        assert result["status"] == "completed"

        # Verify mocks were called correctly
        mock_audio_file.read.assert_called_once()
        mock_client.upload_fileobj.assert_called_once()
        mock_start_transcription_job.assert_called_once()
        mock_wait_for_transcription_job.assert_called_once_with("test-job")
        mock_download_transcript.assert_called_once()
        mock_extract_transcript_text.assert_called_once_with(mock_transcript_data)

    @pytest.mark.asyncio
    @patch("app.aws.transcription.TranscriptionService.transcribe_audio_file")
    async def test_process_voice_success(
        self, mock_transcribe_audio_file, mock_audio_file, transcription_service
    ):
        """Test voice processing with successful transcription."""
        # Set up mock
        mock_transcribe_audio_file.return_value = {
            "transcript": "This is a test transcription.",
            "confidence": 0.97,
            "duration_seconds": 3.5,
            "job_name": "test-job",
            "language_code": "en-US",
            "media_uri": "s3://test-bucket/transcribe/test-job.mp3",
            "status": "completed",
        }

        # Process voice
        file_id = uuid.uuid4()
        result = await transcription_service.process_voice(mock_audio_file, file_id)

        # Verify result
        assert isinstance(result, ProcessedVoice)
        assert result.original_id == file_id
        assert result.processed_text == "This is a test transcription."
        assert result.content_type == "voice_transcript"
        assert result.processing_status == "completed"
        assert result.transcript == "This is a test transcription."
        assert result.confidence_score == 0.97
        assert result.duration_seconds == 3.5

        # Verify mock was called correctly
        mock_transcribe_audio_file.assert_called_once_with(
            file=mock_audio_file, language_code="en-US"
        )

    @pytest.mark.asyncio
    @patch("app.aws.transcription.TranscriptionService.transcribe_audio_file")
    async def test_process_voice_empty_transcript(
        self, mock_transcribe_audio_file, mock_audio_file, transcription_service
    ):
        """Test voice processing with empty transcript."""
        # Set up mock
        mock_transcribe_audio_file.return_value = {
            "transcript": "",
            "confidence": 0.0,
            "duration_seconds": 1.0,
            "job_name": "test-job",
            "language_code": "en-US",
            "media_uri": "s3://test-bucket/transcribe/test-job.mp3",
            "status": "completed",
        }

        # Process voice
        result = await transcription_service.process_voice(mock_audio_file)

        # Verify result
        assert isinstance(result, ProcessedVoice)
        assert result.processing_status == "partial"
        assert result.transcript == "No speech detected in audio"
        assert result.processed_text == "No speech detected in audio"

        # Verify mock was called correctly
        mock_transcribe_audio_file.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.aws.transcription.TranscriptionService.transcribe_audio_file")
    async def test_process_voice_transcribe_error(
        self, mock_transcribe_audio_file, mock_audio_file, transcription_service
    ):
        """Test voice processing with transcription error."""
        # Set up mock to raise error
        mock_transcribe_audio_file.side_effect = TranscribeError(
            message="Transcription failed", operation="transcribe_audio_file"
        )

        # Process voice
        result = await transcription_service.process_voice(mock_audio_file)

        # Verify result
        assert isinstance(result, ProcessedVoice)
        assert result.processing_status == "failed"
        assert "Transcription failed" in result.error_details
        assert result.transcript == "Audio transcription failed"
        assert result.processed_text == "Audio transcription failed"

        # Verify mock was called correctly
        mock_transcribe_audio_file.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.aws.transcription.TranscriptionService.transcribe_audio_file")
    async def test_process_voice_s3_error(
        self, mock_transcribe_audio_file, mock_audio_file, transcription_service
    ):
        """Test voice processing with S3 error."""
        # Set up mock to raise error
        mock_transcribe_audio_file.side_effect = S3Error(
            message="S3 operation failed", operation="upload_file_to_s3"
        )

        # Process voice
        result = await transcription_service.process_voice(mock_audio_file)

        # Verify result
        assert isinstance(result, ProcessedVoice)
        assert result.processing_status == "failed"
        assert "S3 operation failed" in result.error_details
        assert result.transcript == "Audio file processing failed"
        assert result.processed_text == "Audio file processing failed"

        # Verify mock was called correctly
        mock_transcribe_audio_file.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.aws.transcription.TranscriptionService.transcribe_audio_file")
    async def test_process_voice_unexpected_error(
        self, mock_transcribe_audio_file, mock_audio_file, transcription_service
    ):
        """Test voice processing with unexpected error."""
        # Set up mock to raise error
        mock_transcribe_audio_file.side_effect = Exception("Unexpected error")

        # Process voice
        result = await transcription_service.process_voice(mock_audio_file)

        # Verify result
        assert isinstance(result, ProcessedVoice)
        assert result.processing_status == "failed"
        assert "Unexpected error" in result.error_details
        assert result.transcript == "Audio processing failed"
        assert result.processed_text == "Audio processing failed"

        # Verify mock was called correctly
        mock_transcribe_audio_file.assert_called_once()
