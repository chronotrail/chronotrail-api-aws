"""
Speech-to-text transcription service for ChronoTrail API.

This module provides functionality for transcribing voice notes using Amazon Transcribe,
handling various audio formats and quality levels, with proper error handling and
fallback mechanisms.
"""

import asyncio
import io
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from fastapi import HTTPException, UploadFile, status
from pydantic import BaseModel

from app.aws.clients import (
    get_async_s3_client,
    get_async_transcribe_client,
    get_s3_client,
    get_transcribe_client,
    handle_aws_error,
    handle_aws_error_async,
    with_retry,
)
from app.aws.exceptions import FileProcessingError, S3Error, TranscribeError
from app.aws.utils import ALLOWED_AUDIO_TYPES, generate_s3_key
from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import ProcessedVoice

# Configure logger
logger = get_logger(__name__)

# Maximum audio file size for direct API processing (100MB for Transcribe)
MAX_DIRECT_PROCESSING_SIZE = 100 * 1024 * 1024

# Supported audio formats by Amazon Transcribe
TRANSCRIBE_SUPPORTED_FORMATS = [
    "audio/mpeg",  # MP3
    "audio/mp4",  # M4A
    "audio/wav",  # WAV
    "audio/x-wav",  # WAV alternative
    "audio/ogg",  # OGG
    "audio/webm",  # WEBM
    "audio/flac",  # FLAC
    "audio/x-flac",  # FLAC alternative
    "audio/amr",  # AMR
]

# Mapping of content types to file extensions
CONTENT_TYPE_TO_EXTENSION = {
    "audio/mpeg": "mp3",
    "audio/mp4": "m4a",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/amr": "amr",
    "audio/aac": "aac",
}

# Default transcription job settings
DEFAULT_TRANSCRIBE_SETTINGS = {
    "language_code": "en-US",
    "max_speaker_labels": 2,
    "show_speaker_labels": True,
    "enable_automatic_punctuation": True,
    "enable_content_redaction": False,
    "vocabulary_name": None,
    "vocabulary_filter_name": None,
}


class TranscriptionService:
    """
    Service for speech-to-text transcription using AWS services.

    This service provides methods for transcribing voice notes using Amazon Transcribe,
    handling various audio formats and quality levels, with proper error handling and
    fallback mechanisms.
    """

    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize the transcription service.

        Args:
            bucket_name: Optional S3 bucket name (defaults to settings.S3_BUCKET_NAME)
        """
        self.bucket_name = bucket_name or settings.S3_BUCKET_NAME
        logger.info(f"Initialized TranscriptionService with bucket: {self.bucket_name}")

    @handle_aws_error(service_name="transcribe")
    def start_transcription_job_sync(
        self,
        job_name: str,
        media_uri: str,
        media_format: str,
        language_code: str = "en-US",
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Start a transcription job using Amazon Transcribe (synchronous).

        Args:
            job_name: Unique name for the transcription job
            media_uri: S3 URI of the audio file
            media_format: Format of the audio file (e.g., 'mp3', 'wav')
            language_code: Language code for transcription
            settings: Optional additional settings for the transcription job

        Returns:
            Dict: Transcription job details

        Raises:
            TranscribeError: If job creation fails
        """
        try:
            # Get Transcribe client
            transcribe_client = get_transcribe_client()

            # Prepare job settings
            job_settings = {
                "ShowSpeakerLabels": True,
                "MaxSpeakerLabels": 2,
                "ChannelIdentification": False,
            }

            # Update with custom settings if provided
            if settings:
                if settings.get("show_speaker_labels") is not None:
                    job_settings["ShowSpeakerLabels"] = settings["show_speaker_labels"]
                if settings.get("max_speaker_labels") is not None:
                    job_settings["MaxSpeakerLabels"] = settings["max_speaker_labels"]
                if settings.get("enable_automatic_punctuation") is not None:
                    job_settings["Settings"] = {
                        "ShowAlternatives": False,
                        "MaxAlternatives": 2,
                    }

            # Start transcription job
            response = transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={"MediaFileUri": media_uri},
                MediaFormat=media_format,
                LanguageCode=language_code,
                Settings={
                    "ShowSpeakerLabels": job_settings["ShowSpeakerLabels"],
                    "MaxSpeakerLabels": job_settings["MaxSpeakerLabels"],
                },
            )

            return response["TranscriptionJob"]

        except Exception as e:
            logger.error(f"Failed to start transcription job: {str(e)}")
            raise TranscribeError(
                message=f"Failed to start transcription job: {str(e)}",
                operation="start_transcription_job_sync",
            )

    @handle_aws_error_async(service_name="transcribe")
    async def start_transcription_job(
        self,
        job_name: str,
        media_uri: str,
        media_format: str,
        language_code: str = "en-US",
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Start a transcription job using Amazon Transcribe (asynchronous).

        Args:
            job_name: Unique name for the transcription job
            media_uri: S3 URI of the audio file
            media_format: Format of the audio file (e.g., 'mp3', 'wav')
            language_code: Language code for transcription
            settings: Optional additional settings for the transcription job

        Returns:
            Dict: Transcription job details

        Raises:
            TranscribeError: If job creation fails
        """
        try:
            # Get async Transcribe client
            transcribe_client_factory = get_async_transcribe_client()

            # Prepare job settings
            job_settings = {
                "ShowSpeakerLabels": True,
                "MaxSpeakerLabels": 2,
                "ChannelIdentification": False,
            }

            # Update with custom settings if provided
            if settings:
                if settings.get("show_speaker_labels") is not None:
                    job_settings["ShowSpeakerLabels"] = settings["show_speaker_labels"]
                if settings.get("max_speaker_labels") is not None:
                    job_settings["MaxSpeakerLabels"] = settings["max_speaker_labels"]
                if settings.get("enable_automatic_punctuation") is not None:
                    job_settings["Settings"] = {
                        "ShowAlternatives": False,
                        "MaxAlternatives": 2,
                    }

            # Start transcription job
            async with transcribe_client_factory() as transcribe_client:
                response = await transcribe_client.start_transcription_job(
                    TranscriptionJobName=job_name,
                    Media={"MediaFileUri": media_uri},
                    MediaFormat=media_format,
                    LanguageCode=language_code,
                    Settings={
                        "ShowSpeakerLabels": job_settings["ShowSpeakerLabels"],
                        "MaxSpeakerLabels": job_settings["MaxSpeakerLabels"],
                    },
                )

            return response["TranscriptionJob"]

        except Exception as e:
            logger.error(f"Failed to start transcription job: {str(e)}")
            raise TranscribeError(
                message=f"Failed to start transcription job: {str(e)}",
                operation="start_transcription_job",
            )

    @handle_aws_error(service_name="transcribe")
    def get_transcription_job_sync(self, job_name: str) -> Dict[str, Any]:
        """
        Get the status and results of a transcription job (synchronous).

        Args:
            job_name: Name of the transcription job

        Returns:
            Dict: Transcription job details

        Raises:
            TranscribeError: If job retrieval fails
        """
        try:
            # Get Transcribe client
            transcribe_client = get_transcribe_client()

            # Get job status
            response = transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )

            return response["TranscriptionJob"]

        except Exception as e:
            logger.error(f"Failed to get transcription job: {str(e)}")
            raise TranscribeError(
                message=f"Failed to get transcription job: {str(e)}",
                operation="get_transcription_job_sync",
            )

    @handle_aws_error_async(service_name="transcribe")
    async def get_transcription_job(self, job_name: str) -> Dict[str, Any]:
        """
        Get the status and results of a transcription job (asynchronous).

        Args:
            job_name: Name of the transcription job

        Returns:
            Dict: Transcription job details

        Raises:
            TranscribeError: If job retrieval fails
        """
        try:
            # Get async Transcribe client
            transcribe_client_factory = get_async_transcribe_client()

            # Get job status
            async with transcribe_client_factory() as transcribe_client:
                response = await transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )

            return response["TranscriptionJob"]

        except Exception as e:
            logger.error(f"Failed to get transcription job: {str(e)}")
            raise TranscribeError(
                message=f"Failed to get transcription job: {str(e)}",
                operation="get_transcription_job",
            )

    async def wait_for_transcription_job(
        self, job_name: str, timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Wait for a transcription job to complete with retry logic.

        Args:
            job_name: Name of the transcription job
            timeout: Maximum time to wait in seconds

        Returns:
            Dict: Completed transcription job details

        Raises:
            TranscribeError: If job fails or times out
        """
        start_time = time.time()

        while True:
            # Check if timeout has been reached
            if time.time() - start_time > timeout:
                raise TranscribeError(
                    message=f"Transcription job timed out after {timeout} seconds",
                    operation="wait_for_transcription_job",
                )

            # Get job status
            job = await self.get_transcription_job(job_name)
            status = job.get("TranscriptionJobStatus")

            # Check if job is complete
            if status == "COMPLETED":
                return job

            # Check if job failed
            if status == "FAILED":
                error_message = job.get("FailureReason", "Unknown error")
                raise TranscribeError(
                    message=f"Transcription job failed: {error_message}",
                    operation="wait_for_transcription_job",
                )

            # Wait before checking again
            await asyncio.sleep(5)

    async def download_transcript(self, transcript_uri: str) -> Dict[str, Any]:
        """
        Download and parse a transcript file from S3.

        Args:
            transcript_uri: URI of the transcript file

        Returns:
            Dict: Parsed transcript data

        Raises:
            S3Error: If download fails
            TranscribeError: If parsing fails
        """
        try:
            # Extract bucket and key from URI
            # URI format: s3://bucket-name/path/to/transcript.json
            parts = transcript_uri.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1]

            # Get S3 client
            s3_client_factory = get_async_s3_client()

            # Download transcript file
            file_obj = io.BytesIO()
            async with s3_client_factory() as s3_client:
                await s3_client.download_fileobj(bucket, key, file_obj)

            # Parse transcript
            file_obj.seek(0)
            transcript_data = json.loads(file_obj.read().decode("utf-8"))

            return transcript_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse transcript: {str(e)}")
            raise TranscribeError(
                message=f"Failed to parse transcript: {str(e)}",
                operation="download_transcript",
            )
        except Exception as e:
            logger.error(f"Failed to download transcript: {str(e)}")
            raise S3Error(
                message=f"Failed to download transcript: {str(e)}",
                operation="download_transcript",
            )

    def extract_transcript_text(
        self, transcript_data: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Extract transcript text and confidence score from transcript data.

        Args:
            transcript_data: Parsed transcript data

        Returns:
            Tuple[str, float]: Transcript text and average confidence score

        Raises:
            TranscribeError: If extraction fails
        """
        try:
            # Extract transcript text
            results = transcript_data.get("results", {})
            transcripts = results.get("transcripts", [])

            if not transcripts:
                return "", 0.0

            transcript_text = transcripts[0].get("transcript", "")

            # Calculate average confidence score
            items = results.get("items", [])
            if not items:
                return transcript_text, 0.0

            confidence_sum = 0.0
            confidence_count = 0

            for item in items:
                # Skip punctuation items as they have 0 confidence and skew the average
                if item.get("type") == "punctuation":
                    continue

                alternatives = item.get("alternatives", [])
                if alternatives:
                    confidence = alternatives[0].get("confidence")
                    if confidence is not None:
                        confidence_sum += float(confidence)
                        confidence_count += 1

            avg_confidence = (
                confidence_sum / confidence_count if confidence_count > 0 else 0.0
            )

            return transcript_text, avg_confidence

        except Exception as e:
            logger.error(f"Failed to extract transcript text: {str(e)}")
            raise TranscribeError(
                message=f"Failed to extract transcript text: {str(e)}",
                operation="extract_transcript_text",
            )

    async def transcribe_audio_file(
        self,
        file: Union[UploadFile, bytes, str],
        job_name: Optional[str] = None,
        language_code: str = "en-US",
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file using Amazon Transcribe.

        This method handles the complete transcription workflow:
        1. Upload the file to S3 (if not already there)
        2. Start a transcription job
        3. Wait for the job to complete
        4. Download and parse the transcript

        Args:
            file: Audio file as UploadFile, raw bytes, or S3 URI
            job_name: Optional name for the transcription job
            language_code: Language code for transcription
            settings: Optional additional settings for the transcription job

        Returns:
            Dict: Transcription results

        Raises:
            TranscribeError: If transcription fails
            S3Error: If file upload fails
        """
        try:
            # Generate a unique job name if not provided
            if not job_name:
                job_name = f"transcribe-{uuid.uuid4()}"

            # Handle different file input types
            if isinstance(file, str) and file.startswith("s3://"):
                # File is already in S3
                media_uri = file
                # Extract format from URI
                file_extension = os.path.splitext(media_uri)[1].lower().lstrip(".")
                media_format = file_extension
            else:
                # Upload file to S3
                if isinstance(file, UploadFile):
                    # Read file content
                    content = await file.read()
                    await file.seek(0)  # Reset file pointer

                    # Determine file format
                    content_type = file.content_type
                    file_extension = CONTENT_TYPE_TO_EXTENSION.get(
                        content_type,
                        os.path.splitext(file.filename)[1].lower().lstrip("."),
                    )

                    # Generate S3 key
                    s3_key = f"transcribe/{job_name}.{file_extension}"

                    # Upload to S3
                    s3_client_factory = get_async_s3_client()
                    async with s3_client_factory() as s3_client:
                        await s3_client.upload_fileobj(
                            io.BytesIO(content), self.bucket_name, s3_key
                        )
                else:
                    # Raw bytes
                    content = file
                    file_extension = "mp3"  # Default to mp3 if unknown
                    s3_key = f"transcribe/{job_name}.{file_extension}"

                    # Upload to S3
                    s3_client_factory = get_async_s3_client()
                    async with s3_client_factory() as s3_client:
                        await s3_client.upload_fileobj(
                            io.BytesIO(content), self.bucket_name, s3_key
                        )

                # Create S3 URI
                media_uri = f"s3://{self.bucket_name}/{s3_key}"
                media_format = file_extension

            # Start transcription job
            job = await self.start_transcription_job(
                job_name=job_name,
                media_uri=media_uri,
                media_format=media_format,
                language_code=language_code,
                settings=settings,
            )

            # Wait for job to complete
            completed_job = await self.wait_for_transcription_job(job_name)

            # Handle case where completed_job is a coroutine (for testing)
            if hasattr(completed_job, "__await__"):
                completed_job = await completed_job

            # Download transcript
            transcript_uri = completed_job.get("Transcript", {}).get(
                "TranscriptFileUri"
            )
            if not transcript_uri:
                raise TranscribeError(
                    message="Transcript URI not found in completed job",
                    operation="transcribe_audio_file",
                )

            transcript_data = await self.download_transcript(transcript_uri)

            # Extract transcript text and confidence
            transcript_text, confidence = self.extract_transcript_text(transcript_data)

            # Calculate duration if available
            duration_seconds = None
            if "results" in transcript_data:
                items = transcript_data["results"].get("items", [])
                if items:
                    last_item = items[-1]
                    if "end_time" in last_item:
                        duration_seconds = float(last_item["end_time"])

            return {
                "transcript": transcript_text,
                "confidence": confidence,
                "duration_seconds": duration_seconds,
                "job_name": job_name,
                "language_code": language_code,
                "media_uri": media_uri,
                "status": "completed",
            }

        except (TranscribeError, S3Error) as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during transcription: {str(e)}")
            raise TranscribeError(
                message=f"Unexpected error during transcription: {str(e)}",
                operation="transcribe_audio_file",
            )

    async def process_voice(
        self,
        file: Union[UploadFile, bytes],
        original_id: Optional[UUID] = None,
        language_code: str = "en-US",
    ) -> ProcessedVoice:
        """
        Process a voice recording using speech-to-text with fallback mechanisms.

        This method attempts to transcribe a voice recording and handles errors gracefully.

        Args:
            file: Voice recording as UploadFile or raw bytes
            original_id: Optional ID of the original file record
            language_code: Language code for transcription

        Returns:
            ProcessedVoice: Processing results

        Raises:
            FileProcessingError: If processing fails completely
        """
        # Initialize result
        result = ProcessedVoice(
            original_id=original_id or UUID("00000000-0000-0000-0000-000000000000"),
            processed_text="",
            content_type="voice_transcript",
            processing_status="completed",
            transcript="",
            confidence_score=0.0,
            duration_seconds=None,
        )

        try:
            # Transcribe audio file
            transcription = await self.transcribe_audio_file(
                file=file, language_code=language_code
            )

            # Update result with transcription data
            result.transcript = transcription["transcript"]
            result.processed_text = transcription["transcript"]
            result.confidence_score = transcription["confidence"]
            result.duration_seconds = transcription["duration_seconds"]

            # Handle empty transcript
            if not result.transcript.strip():
                result.processing_status = "partial"
                result.transcript = "No speech detected in audio"
                result.processed_text = "No speech detected in audio"

            return result

        except TranscribeError as e:
            logger.warning(f"Transcription failed: {str(e)}")
            result.processing_status = "failed"
            result.error_details = f"Transcription failed: {str(e)}"
            result.transcript = "Audio transcription failed"
            result.processed_text = "Audio transcription failed"

            return result

        except S3Error as e:
            logger.warning(f"S3 operation failed during transcription: {str(e)}")
            result.processing_status = "failed"
            result.error_details = f"S3 operation failed: {str(e)}"
            result.transcript = "Audio file processing failed"
            result.processed_text = "Audio file processing failed"

            return result

        except Exception as e:
            logger.error(f"Unexpected error during voice processing: {str(e)}")
            result.processing_status = "failed"
            result.error_details = f"Unexpected error: {str(e)}"
            result.transcript = "Audio processing failed"
            result.processed_text = "Audio processing failed"

            return result


# Create a singleton instance
transcription_service = TranscriptionService()
