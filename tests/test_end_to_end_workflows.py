"""
End-to-end tests for ChronoTrail API user workflows.

This module contains comprehensive end-to-end tests that simulate complete user journeys
from data submission to querying, conversational media retrieval flows, and location
tagging and search functionality.
"""

import io
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
from fastapi import status
from httpx import AsyncClient
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import DailyUsage as DBDailyUsage
from app.models.database import LocationVisit as DBLocationVisit
from app.models.database import MediaFile as DBMediaFile
from app.models.database import QuerySession as DBQuerySession
from app.models.database import TextNote as DBTextNote
from app.models.database import User as DBUser
from app.models.schemas import (
    ContentType,
    FileType,
    LocationVisitCreate,
    MediaReference,
    OAuthProvider,
    ProcessedPhoto,
    ProcessedVoice,
    QueryRequest,
    QueryResponse,
    SubscriptionTier,
    TextNoteCreate,
    User,
)


# Mock all the services to avoid database and AWS calls
@pytest.fixture(autouse=True)
def mock_all_services(mock_auth_user):
    """Mock all services to avoid external dependencies."""

    async def mock_validate_content_limits(request, current_user, db):
        return current_user

    async def mock_validate_query_limits(request, current_user, db):
        return current_user

    with (
        patch(
            "app.services.usage.UsageService.check_daily_content_limit",
            return_value=None,
        ),
        patch(
            "app.services.usage.UsageService.increment_content_usage", return_value=None
        ),
        patch(
            "app.services.usage.UsageService.increment_query_usage", return_value=None
        ),
        patch(
            "app.services.usage.UsageService.validate_query_date_range",
            return_value=True,
        ),
        patch(
            "app.middleware.usage_validation.UsageLimitMiddleware.validate_content_limits",
            side_effect=mock_validate_content_limits,
        ),
        patch(
            "app.middleware.usage_validation.UsageLimitMiddleware.validate_query_limits",
            side_effect=mock_validate_query_limits,
        ),
    ):
        yield


class TestCompleteUserJourneys:
    """Test complete user journeys from data submission to querying."""

    def create_test_image(self, size=(100, 100), format="JPEG"):
        """Create a test image file."""
        image = Image.new("RGB", size, color="red")
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=format)
        img_bytes.seek(0)
        return img_bytes

    def create_test_audio(self):
        """Create a test audio file."""
        audio_data = b"fake_audio_data" * 1000
        return io.BytesIO(audio_data)

    @pytest.mark.asyncio
    async def test_complete_timeline_creation_and_query_journey(
        self, client: AsyncClient, mock_db: AsyncSession, mock_auth_user: Dict[str, Any]
    ):
        """
        Test complete user journey: Create location visits, add text notes,
        upload photos and voice notes, then query the timeline.
        """
        user_id = mock_auth_user["user"].id
        auth_header = {"Authorization": f"Bearer {mock_auth_user['token']}"}

        # Step 1: Create a location visit
        visit_id = uuid4()
        db_visit = DBLocationVisit(
            id=visit_id,
            user_id=user_id,
            longitude=-122.4194,
            latitude=37.7749,
            address="Golden Gate Park, San Francisco, CA",
            names=["Golden Gate Park", "Park"],
            visit_time=datetime.utcnow() - timedelta(hours=2),
            duration=120,
            description="Beautiful morning walk in the park",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        with patch(
            "app.repositories.location_visits.location_visit_repository.create",
            return_value=db_visit,
        ):
            location_response = await client.post(
                "/api/v1/locations",
                json={
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "Golden Gate Park, San Francisco, CA",
                    "names": ["Golden Gate Park", "Park"],
                    "visit_time": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                    "duration": 120,
                    "description": "Beautiful morning walk in the park",
                },
                headers=auth_header,
            )

            assert location_response.status_code == status.HTTP_201_CREATED
            location_data = location_response.json()
            assert location_data["id"] == str(visit_id)

        # Step 2: Add a text note about the visit
        note_id = uuid4()
        db_note = DBTextNote(
            id=note_id,
            user_id=user_id,
            text_content="Saw beautiful cherry blossoms and took some photos. The weather was perfect for a morning walk.",
            timestamp=datetime.utcnow() - timedelta(hours=1, minutes=30),
            longitude=-122.4194,
            latitude=37.7749,
            address="Golden Gate Park, San Francisco, CA",
            names=["Golden Gate Park"],
            created_at=datetime.utcnow(),
        )

        with (
            patch(
                "app.repositories.text_notes.text_note_repository.create",
                return_value=db_note,
            ),
            patch(
                "app.aws.embedding.embedding_service.process_and_store_text",
                return_value="doc_note_123",
            ),
            patch("app.services.usage.UsageService.increment_content_usage"),
        ):

            note_response = await client.post(
                "/api/v1/notes",
                json={
                    "text_content": "Saw beautiful cherry blossoms and took some photos. The weather was perfect for a morning walk.",
                    "timestamp": (
                        datetime.utcnow() - timedelta(hours=1, minutes=30)
                    ).isoformat(),
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "Golden Gate Park, San Francisco, CA",
                    "names": ["Golden Gate Park"],
                },
                headers=auth_header,
            )

            assert note_response.status_code == status.HTTP_201_CREATED
            note_data = note_response.json()
            assert note_data["id"] == str(note_id)

        # Step 3: Upload a photo from the visit
        photo_id = uuid4()
        test_image = self.create_test_image()

        db_photo = DBMediaFile(
            id=photo_id,
            user_id=user_id,
            file_type="photo",
            file_path="photos/cherry_blossoms.jpg",
            original_filename="cherry_blossoms.jpg",
            file_size=1024000,
            timestamp=datetime.utcnow() - timedelta(hours=1, minutes=15),
            longitude=-122.4194,
            latitude=37.7749,
            address="Golden Gate Park, San Francisco, CA",
            names=["Golden Gate Park"],
            created_at=datetime.utcnow(),
        )

        processed_photo = ProcessedPhoto(
            original_id=photo_id,
            processed_text="Beautiful cherry blossoms in full bloom at Golden Gate Park",
            content_type="image_desc",
            processing_status="completed",
            extracted_text="",
            image_description="Beautiful cherry blossoms in full bloom at Golden Gate Park",
            detected_objects=["tree", "flowers", "park", "path"],
        )

        async def mock_upload_file(*args, **kwargs):
            return {
                "file_id": str(photo_id),
                "s3_key": "photos/cherry_blossoms.jpg",
                "file_size": 1024000,
                "timestamp": datetime.utcnow(),
            }

        with (
            patch(
                "app.aws.storage.file_storage_service.upload_file",
                side_effect=mock_upload_file,
            ),
            patch(
                "app.repositories.media_files.media_file_repository.create",
                return_value=db_photo,
            ),
            patch(
                "app.aws.image_processing.image_processing_service.process_image",
                return_value=processed_photo,
            ),
            patch(
                "app.aws.embedding.embedding_service.process_and_store_text",
                return_value="doc_photo_123",
            ),
            patch("app.services.usage.UsageService.increment_content_usage"),
        ):

            photo_response = await client.post(
                "/api/v1/photos",
                files={"file": ("cherry_blossoms.jpg", test_image, "image/jpeg")},
                data={
                    "timestamp": (
                        datetime.utcnow() - timedelta(hours=1, minutes=15)
                    ).isoformat(),
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "Golden Gate Park, San Francisco, CA",
                    "names": "Golden Gate Park",
                },
                headers=auth_header,
            )

            assert photo_response.status_code == status.HTTP_201_CREATED
            photo_data = photo_response.json()
            assert photo_data["processing_status"] == "completed"

        # Step 4: Upload a voice note
        voice_id = uuid4()
        test_audio = self.create_test_audio()

        db_voice = DBMediaFile(
            id=voice_id,
            user_id=user_id,
            file_type="voice",
            file_path="voice/park_thoughts.mp3",
            original_filename="park_thoughts.mp3",
            file_size=50000,
            timestamp=datetime.utcnow() - timedelta(hours=1),
            longitude=-122.4194,
            latitude=37.7749,
            address="Golden Gate Park, San Francisco, CA",
            names=["Golden Gate Park"],
            created_at=datetime.utcnow(),
        )

        processed_voice = ProcessedVoice(
            original_id=voice_id,
            processed_text="This park is absolutely beautiful today. The cherry blossoms are in full bloom and there are so many people enjoying the sunshine.",
            content_type="voice_transcript",
            processing_status="completed",
            transcript="This park is absolutely beautiful today. The cherry blossoms are in full bloom and there are so many people enjoying the sunshine.",
            confidence_score=0.95,
            duration_seconds=15.5,
        )

        async def mock_upload_voice_file(*args, **kwargs):
            return {
                "file_id": str(voice_id),
                "s3_key": "voice/park_thoughts.mp3",
                "file_size": 50000,
                "timestamp": datetime.utcnow(),
            }

        with (
            patch(
                "app.aws.storage.file_storage_service.upload_file",
                side_effect=mock_upload_voice_file,
            ),
            patch(
                "app.repositories.media_files.media_file_repository.create",
                return_value=db_voice,
            ),
            patch(
                "app.aws.transcription.transcription_service.process_voice",
                return_value=processed_voice,
            ),
            patch(
                "app.aws.embedding.embedding_service.process_and_store_text",
                return_value="doc_voice_123",
            ),
            patch("app.services.usage.UsageService.increment_content_usage"),
        ):

            voice_response = await client.post(
                "/api/v1/voice",
                files={"file": ("park_thoughts.mp3", test_audio, "audio/mpeg")},
                data={
                    "timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "Golden Gate Park, San Francisco, CA",
                    "names": "Golden Gate Park",
                },
                headers=auth_header,
            )

            assert voice_response.status_code == status.HTTP_201_CREATED
            voice_data = voice_response.json()
            assert voice_data["processing_status"] == "completed"

        # Step 5: Query the timeline about the park visit
        query_response_data = QueryResponse(
            answer="Based on your timeline, you visited Golden Gate Park this morning for about 2 hours. You took a beautiful walk and noted that the cherry blossoms were in full bloom. You captured some photos of the blossoms and recorded a voice note about how beautiful the park was with all the people enjoying the sunshine.",
            sources=[
                {
                    "type": "location_visit",
                    "content": "Beautiful morning walk in the park",
                    "timestamp": datetime.utcnow() - timedelta(hours=2),
                    "relevance_score": 0.95,
                    "location": {"address": "Golden Gate Park, San Francisco, CA"},
                },
                {
                    "type": "text_note",
                    "content": "Saw beautiful cherry blossoms and took some photos. The weather was perfect for a morning walk.",
                    "timestamp": datetime.utcnow() - timedelta(hours=1, minutes=30),
                    "relevance_score": 0.92,
                },
                {
                    "type": "voice_transcript",
                    "content": "This park is absolutely beautiful today. The cherry blossoms are in full bloom and there are so many people enjoying the sunshine.",
                    "timestamp": datetime.utcnow() - timedelta(hours=1),
                    "relevance_score": 0.90,
                },
            ],
            media_references=[
                MediaReference(
                    media_id=photo_id,
                    media_type="photo",
                    description="Photo of cherry blossoms at Golden Gate Park",
                    timestamp=datetime.utcnow() - timedelta(hours=1, minutes=15),
                    location={"address": "Golden Gate Park, San Francisco, CA"},
                ),
                MediaReference(
                    media_id=voice_id,
                    media_type="voice",
                    description="Voice note about the beautiful park",
                    timestamp=datetime.utcnow() - timedelta(hours=1),
                    location={"address": "Golden Gate Park, San Francisco, CA"},
                ),
            ],
            session_id="session_123",
        )

        with (
            patch(
                "app.services.query.query_processing_service.process_query",
                return_value=query_response_data,
            ),
            patch("app.services.usage.UsageService.increment_query_usage"),
        ):

            query_response = await client.post(
                "/api/v1/query/",
                json={
                    "query": "What did I do at the park today?",
                    "session_id": "session_123",
                },
                headers=auth_header,
            )

            assert query_response.status_code == status.HTTP_200_OK
            query_data = query_response.json()
            assert "Golden Gate Park" in query_data["answer"]
            assert "cherry blossoms" in query_data["answer"]
            assert len(query_data["sources"]) == 3
            assert len(query_data["media_references"]) == 2
            assert query_data["session_id"] == "session_123"

        # Step 6: Retrieve the referenced media files
        # Get photo
        fake_image_data = b"fake_image_data" * 1000

        with (
            patch(
                "app.repositories.media_files.media_file_repository.get_by_user",
                return_value=db_photo,
            ),
            patch(
                "app.aws.storage.file_storage_service.download_file",
                return_value=fake_image_data,
            ),
        ):

            photo_retrieval_response = await client.get(
                f"/api/v1/media/{photo_id}", headers=auth_header
            )

            assert photo_retrieval_response.status_code == status.HTTP_200_OK
            assert photo_retrieval_response.headers["content-type"] == "image/jpeg"
            assert photo_retrieval_response.content == fake_image_data

        # Get voice note
        fake_audio_data = b"fake_audio_data" * 1000

        with (
            patch(
                "app.repositories.media_files.media_file_repository.get_by_user",
                return_value=db_voice,
            ),
            patch(
                "app.aws.storage.file_storage_service.download_file",
                return_value=fake_audio_data,
            ),
        ):

            voice_retrieval_response = await client.get(
                f"/api/v1/media/{voice_id}", headers=auth_header
            )

            assert voice_retrieval_response.status_code == status.HTTP_200_OK
            assert voice_retrieval_response.headers["content-type"] == "audio/mpeg"
            assert voice_retrieval_response.content == fake_audio_data

    @pytest.mark.asyncio
    async def test_multi_day_timeline_with_location_search(
        self, client: AsyncClient, mock_db: AsyncSession, mock_auth_user: Dict[str, Any]
    ):
        """
        Test creating timeline data across multiple days and locations,
        then searching by location and time.
        """
        user_id = mock_auth_user["user"].id
        auth_header = {"Authorization": f"Bearer {mock_auth_user['token']}"}

        # Create visits across multiple days and locations
        locations_data = [
            {
                "id": uuid4(),
                "longitude": -122.4194,
                "latitude": 37.7749,
                "address": "Golden Gate Park, San Francisco, CA",
                "names": ["Golden Gate Park", "Park"],
                "visit_time": datetime.utcnow() - timedelta(days=2),
                "description": "Morning jog in the park",
            },
            {
                "id": uuid4(),
                "longitude": -122.4089,
                "latitude": 37.7853,
                "address": "Blue Bottle Coffee, San Francisco, CA",
                "names": ["Blue Bottle Coffee", "Coffee Shop"],
                "visit_time": datetime.utcnow() - timedelta(days=1),
                "description": "Great coffee meeting with Sarah",
            },
            {
                "id": uuid4(),
                "longitude": -122.3959,
                "latitude": 37.7937,
                "address": "Ferry Building, San Francisco, CA",
                "names": ["Ferry Building", "Market"],
                "visit_time": datetime.utcnow() - timedelta(hours=3),
                "description": "Farmers market visit",
            },
        ]

        # Create all location visits
        for location in locations_data:
            db_visit = DBLocationVisit(
                id=location["id"],
                user_id=user_id,
                longitude=location["longitude"],
                latitude=location["latitude"],
                address=location["address"],
                names=location["names"],
                visit_time=location["visit_time"],
                description=location["description"],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            with patch(
                "app.repositories.location_visits.location_visit_repository.create",
                return_value=db_visit,
            ):
                response = await client.post(
                    "/api/v1/locations",
                    json={
                        "longitude": location["longitude"],
                        "latitude": location["latitude"],
                        "address": location["address"],
                        "names": location["names"],
                        "visit_time": location["visit_time"].isoformat(),
                        "description": location["description"],
                    },
                    headers=auth_header,
                )

                assert response.status_code == status.HTTP_201_CREATED

        # Query for locations from the last 2 days
        mock_visits = [
            DBLocationVisit(
                id=loc["id"],
                user_id=user_id,
                longitude=loc["longitude"],
                latitude=loc["latitude"],
                address=loc["address"],
                names=loc["names"],
                visit_time=loc["visit_time"],
                description=loc["description"],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            for loc in locations_data
        ]

        with (
            patch(
                "app.repositories.location_visits.location_visit_repository.get_by_user_with_date_range",
                return_value=mock_visits,
            ),
            patch(
                "app.repositories.location_visits.location_visit_repository.count_by_user",
                return_value=3,
            ),
        ):

            start_date = (datetime.utcnow() - timedelta(days=2)).date()
            end_date = datetime.utcnow().date()

            locations_response = await client.get(
                f"/api/v1/locations?start_date={start_date}&end_date={end_date}",
                headers=auth_header,
            )

            assert locations_response.status_code == status.HTTP_200_OK
            locations_data_response = locations_response.json()
            assert locations_data_response["total"] == 3
            assert len(locations_data_response["visits"]) == 3

            # Verify locations are returned (order may vary in test environment)
            visit_times = [
                visit["visit_time"] for visit in locations_data_response["visits"]
            ]
            assert len(visit_times) == 3  # All visits are present

        # Query for specific location type (coffee shops)
        coffee_query_response = QueryResponse(
            answer="You visited Blue Bottle Coffee yesterday where you had a great coffee meeting with Sarah.",
            sources=[
                {
                    "type": "location_visit",
                    "content": "Great coffee meeting with Sarah",
                    "timestamp": datetime.utcnow() - timedelta(days=1),
                    "relevance_score": 0.95,
                    "location": {"address": "Blue Bottle Coffee, San Francisco, CA"},
                }
            ],
            media_references=[],
            session_id="session_coffee_123",
        )

        with patch(
            "app.services.query.query_processing_service.process_query",
            return_value=coffee_query_response,
        ):
            query_response = await client.post(
                "/api/v1/query/",
                json={
                    "query": "Where did I go for coffee recently?",
                    "session_id": "session_coffee_123",
                },
                headers=auth_header,
            )

            assert query_response.status_code == status.HTTP_200_OK
            query_data = query_response.json()
            assert "Blue Bottle Coffee" in query_data["answer"]
            assert "Sarah" in query_data["answer"]


class TestConversationalMediaRetrievalFlows:
    """Test conversational media retrieval flows."""

    def create_test_image(self, size=(100, 100), format="JPEG"):
        """Create a test image file."""
        image = Image.new("RGB", size, color="blue")
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=format)
        img_bytes.seek(0)
        return img_bytes

    @pytest.mark.asyncio
    async def test_conversational_photo_discovery_and_retrieval(
        self, client: AsyncClient, mock_db: AsyncSession, mock_auth_user: Dict[str, Any]
    ):
        """
        Test conversational flow where user asks about photos, gets references,
        and then retrieves specific photos through follow-up queries.
        """
        user_id = mock_auth_user["user"].id
        auth_header = {"Authorization": f"Bearer {mock_auth_user['token']}"}

        # Setup: Create multiple photos from different locations
        photo_ids = [uuid4(), uuid4(), uuid4()]
        photo_data = [
            {
                "id": photo_ids[0],
                "location": "Golden Gate Park",
                "description": "Cherry blossoms in full bloom",
                "timestamp": datetime.utcnow() - timedelta(days=1),
            },
            {
                "id": photo_ids[1],
                "location": "Lombard Street",
                "description": "Famous winding street with flowers",
                "timestamp": datetime.utcnow() - timedelta(days=2),
            },
            {
                "id": photo_ids[2],
                "location": "Fisherman's Wharf",
                "description": "Sea lions on the pier",
                "timestamp": datetime.utcnow() - timedelta(days=3),
            },
        ]

        # Step 1: Initial query about photos
        initial_query_response = QueryResponse(
            answer="You've taken several photos recently! You have photos of cherry blossoms from Golden Gate Park, the famous winding Lombard Street with flowers, and sea lions at Fisherman's Wharf. Would you like me to show you any specific photos?",
            sources=[
                {
                    "type": "image_desc",
                    "content": photo["description"],
                    "timestamp": photo["timestamp"],
                    "relevance_score": 0.9,
                    "location": {"address": f"{photo['location']}, San Francisco, CA"},
                }
                for photo in photo_data
            ],
            media_references=[
                MediaReference(
                    media_id=photo["id"],
                    media_type="photo",
                    description=f"Photo of {photo['description']} at {photo['location']}",
                    timestamp=photo["timestamp"],
                    location={"address": f"{photo['location']}, San Francisco, CA"},
                )
                for photo in photo_data
            ],
            session_id="photo_session_123",
        )

        with patch(
            "app.services.query.query_processing_service.process_query",
            return_value=initial_query_response,
        ):
            initial_response = await client.post(
                "/api/v1/query/",
                json={
                    "query": "What photos have I taken recently?",
                    "session_id": "photo_session_123",
                },
                headers=auth_header,
            )

            assert initial_response.status_code == status.HTTP_200_OK
            initial_data = initial_response.json()
            assert len(initial_data["media_references"]) == 3
            assert "cherry blossoms" in initial_data["answer"]
            assert "Lombard Street" in initial_data["answer"]
            assert "sea lions" in initial_data["answer"]

        # Step 2: Follow-up query asking for specific photo
        followup_query_response = QueryResponse(
            answer="Here's your photo of the cherry blossoms from Golden Gate Park that you took yesterday. The blossoms were in full bloom and looked absolutely beautiful!",
            sources=[
                {
                    "type": "image_desc",
                    "content": "Cherry blossoms in full bloom",
                    "timestamp": datetime.utcnow() - timedelta(days=1),
                    "relevance_score": 0.98,
                    "location": {"address": "Golden Gate Park, San Francisco, CA"},
                }
            ],
            media_references=[
                MediaReference(
                    media_id=photo_ids[0],
                    media_type="photo",
                    description="Photo of cherry blossoms in full bloom at Golden Gate Park",
                    timestamp=datetime.utcnow() - timedelta(days=1),
                    location={"address": "Golden Gate Park, San Francisco, CA"},
                )
            ],
            session_id="photo_session_123",
        )

        with patch(
            "app.services.query.query_processing_service.process_query",
            return_value=followup_query_response,
        ):
            followup_response = await client.post(
                "/api/v1/query/",
                json={
                    "query": "Show me the cherry blossom photo",
                    "session_id": "photo_session_123",
                },
                headers=auth_header,
            )

            assert followup_response.status_code == status.HTTP_200_OK
            followup_data = followup_response.json()
            assert len(followup_data["media_references"]) == 1
            assert followup_data["media_references"][0]["media_id"] == str(photo_ids[0])
            assert "cherry blossoms" in followup_data["answer"]

        # Step 3: Retrieve the actual photo file
        db_photo = DBMediaFile(
            id=photo_ids[0],
            user_id=user_id,
            file_type="photo",
            file_path="photos/cherry_blossoms.jpg",
            original_filename="cherry_blossoms.jpg",
            file_size=1024000,
            timestamp=datetime.utcnow() - timedelta(days=1),
            longitude=-122.4194,
            latitude=37.7749,
            address="Golden Gate Park, San Francisco, CA",
            names=["Golden Gate Park"],
            created_at=datetime.utcnow(),
        )

        fake_image_data = b"fake_cherry_blossom_image_data" * 100

        with (
            patch(
                "app.repositories.media_files.media_file_repository.get_by_user",
                return_value=db_photo,
            ),
            patch(
                "app.aws.storage.file_storage_service.download_file",
                return_value=fake_image_data,
            ),
        ):

            photo_response = await client.get(
                f"/api/v1/media/{photo_ids[0]}", headers=auth_header
            )

            assert photo_response.status_code == status.HTTP_200_OK
            assert photo_response.headers["content-type"] == "image/jpeg"
            assert photo_response.content == fake_image_data

        # Step 4: Get photo info for context
        with patch(
            "app.repositories.media_files.media_file_repository.get_by_user",
            return_value=db_photo,
        ):
            info_response = await client.get(
                f"/api/v1/media/{photo_ids[0]}/info", headers=auth_header
            )

            assert info_response.status_code == status.HTTP_200_OK
            info_data = info_response.json()
            assert info_data["id"] == str(photo_ids[0])
            assert (
                info_data["location"]["address"]
                == "Golden Gate Park, San Francisco, CA"
            )
            assert info_data["location"]["names"] == ["Golden Gate Park"]

    @pytest.mark.asyncio
    async def test_voice_note_conversation_flow(
        self, client: AsyncClient, mock_db: AsyncSession, mock_auth_user: Dict[str, Any]
    ):
        """
        Test conversational flow with voice notes including transcription
        context and follow-up questions.
        """
        user_id = mock_auth_user["user"].id
        auth_header = {"Authorization": f"Bearer {mock_auth_user['token']}"}

        # Setup: Create voice notes from a meeting
        voice_ids = [uuid4(), uuid4()]

        # Step 1: Query about voice notes from a meeting
        meeting_query_response = QueryResponse(
            answer="You recorded two voice notes during your meeting at Blue Bottle Coffee yesterday. In the first note, you mentioned discussing the new project timeline with Sarah and agreeing on the deliverables. In the second note, you recorded action items including following up with the design team and scheduling the next review meeting.",
            sources=[
                {
                    "type": "voice_transcript",
                    "content": "Had a great discussion with Sarah about the new project timeline. We agreed on the key deliverables and she's excited about the direction we're taking.",
                    "timestamp": datetime.utcnow() - timedelta(days=1, hours=2),
                    "relevance_score": 0.95,
                },
                {
                    "type": "voice_transcript",
                    "content": "Action items from the meeting: follow up with design team about mockups, schedule next review meeting for Friday, and send Sarah the updated project brief.",
                    "timestamp": datetime.utcnow()
                    - timedelta(days=1, hours=1, minutes=45),
                    "relevance_score": 0.93,
                },
            ],
            media_references=[
                MediaReference(
                    media_id=voice_ids[0],
                    media_type="voice",
                    description="Voice note about project discussion with Sarah",
                    timestamp=datetime.utcnow() - timedelta(days=1, hours=2),
                    location={"address": "Blue Bottle Coffee, San Francisco, CA"},
                ),
                MediaReference(
                    media_id=voice_ids[1],
                    media_type="voice",
                    description="Voice note with meeting action items",
                    timestamp=datetime.utcnow()
                    - timedelta(days=1, hours=1, minutes=45),
                    location={"address": "Blue Bottle Coffee, San Francisco, CA"},
                ),
            ],
            session_id="meeting_session_123",
        )

        with patch(
            "app.services.query.query_processing_service.process_query",
            return_value=meeting_query_response,
        ):
            meeting_response = await client.post(
                "/api/v1/query/",
                json={
                    "query": "What voice notes did I record during my meeting yesterday?",
                    "session_id": "meeting_session_123",
                },
                headers=auth_header,
            )

            assert meeting_response.status_code == status.HTTP_200_OK
            meeting_data = meeting_response.json()
            assert len(meeting_data["media_references"]) == 2
            assert "Sarah" in meeting_data["answer"]
            assert "action items" in meeting_data["answer"]

        # Step 2: Follow-up query about specific action items
        action_items_response = QueryResponse(
            answer="Based on your voice note, your action items from the meeting were: 1) Follow up with the design team about mockups, 2) Schedule the next review meeting for Friday, and 3) Send Sarah the updated project brief.",
            sources=[
                {
                    "type": "voice_transcript",
                    "content": "Action items from the meeting: follow up with design team about mockups, schedule next review meeting for Friday, and send Sarah the updated project brief.",
                    "timestamp": datetime.utcnow()
                    - timedelta(days=1, hours=1, minutes=45),
                    "relevance_score": 0.98,
                }
            ],
            media_references=[
                MediaReference(
                    media_id=voice_ids[1],
                    media_type="voice",
                    description="Voice note with meeting action items",
                    timestamp=datetime.utcnow()
                    - timedelta(days=1, hours=1, minutes=45),
                    location={"address": "Blue Bottle Coffee, San Francisco, CA"},
                )
            ],
            session_id="meeting_session_123",
        )

        with patch(
            "app.services.query.query_processing_service.process_query",
            return_value=action_items_response,
        ):
            action_response = await client.post(
                "/api/v1/query/",
                json={
                    "query": "What were the action items from that meeting?",
                    "session_id": "meeting_session_123",
                },
                headers=auth_header,
            )

            assert action_response.status_code == status.HTTP_200_OK
            action_data = action_response.json()
            assert "design team" in action_data["answer"]
            assert "Friday" in action_data["answer"]
            assert "project brief" in action_data["answer"]

        # Step 3: Retrieve the actual voice note
        db_voice = DBMediaFile(
            id=voice_ids[1],
            user_id=user_id,
            file_type="voice",
            file_path="voice/meeting_actions.mp3",
            original_filename="meeting_actions.mp3",
            file_size=75000,
            timestamp=datetime.utcnow() - timedelta(days=1, hours=1, minutes=45),
            longitude=-122.4089,
            latitude=37.7853,
            address="Blue Bottle Coffee, San Francisco, CA",
            names=["Blue Bottle Coffee"],
            created_at=datetime.utcnow(),
        )

        fake_audio_data = b"fake_meeting_audio_data" * 200

        with (
            patch(
                "app.repositories.media_files.media_file_repository.get_by_user",
                return_value=db_voice,
            ),
            patch(
                "app.aws.storage.file_storage_service.download_file",
                return_value=fake_audio_data,
            ),
        ):

            voice_response = await client.get(
                f"/api/v1/media/{voice_ids[1]}", headers=auth_header
            )

            assert voice_response.status_code == status.HTTP_200_OK
            assert voice_response.headers["content-type"] == "audio/mpeg"
            assert voice_response.content == fake_audio_data


class TestLocationTaggingAndSearchFunctionality:
    """Test location tagging and search functionality."""

    @pytest.mark.asyncio
    async def test_location_tagging_and_name_based_search(
        self, client: AsyncClient, mock_db: AsyncSession, mock_auth_user: Dict[str, Any]
    ):
        """
        Test creating locations with multiple names/tags and searching by those tags.
        """
        user_id = mock_auth_user["user"].id
        auth_header = {"Authorization": f"Bearer {mock_auth_user['token']}"}

        # Step 1: Create locations with multiple names/tags
        locations_with_tags = [
            {
                "id": uuid4(),
                "longitude": -122.4194,
                "latitude": 37.7749,
                "address": "Golden Gate Park, San Francisco, CA",
                "names": ["Golden Gate Park", "Park", "Nature", "Running Spot"],
                "description": "My favorite running location",
            },
            {
                "id": uuid4(),
                "longitude": -122.4089,
                "latitude": 37.7853,
                "address": "Blue Bottle Coffee, San Francisco, CA",
                "names": [
                    "Blue Bottle Coffee",
                    "Coffee Shop",
                    "Work Spot",
                    "Meeting Place",
                ],
                "description": "Great place for work meetings",
            },
            {
                "id": uuid4(),
                "longitude": -122.3959,
                "latitude": 37.7937,
                "address": "Ferry Building, San Francisco, CA",
                "names": ["Ferry Building", "Market", "Food", "Weekend Spot"],
                "description": "Saturday farmers market",
            },
        ]

        # Create all locations
        for location in locations_with_tags:
            db_visit = DBLocationVisit(
                id=location["id"],
                user_id=user_id,
                longitude=location["longitude"],
                latitude=location["latitude"],
                address=location["address"],
                names=location["names"],
                visit_time=datetime.utcnow() - timedelta(hours=2),
                description=location["description"],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            with patch(
                "app.repositories.location_visits.location_visit_repository.create",
                return_value=db_visit,
            ):
                response = await client.post(
                    "/api/v1/locations",
                    json={
                        "longitude": location["longitude"],
                        "latitude": location["latitude"],
                        "address": location["address"],
                        "names": location["names"],
                        "visit_time": (
                            datetime.utcnow() - timedelta(hours=2)
                        ).isoformat(),
                        "description": location["description"],
                    },
                    headers=auth_header,
                )

                assert response.status_code == status.HTTP_201_CREATED
                data = response.json()
                assert set(data["names"]) == set(location["names"])

        # Step 2: Search by specific tag - "Work Spot"
        work_spot_query = QueryResponse(
            answer="You've tagged Blue Bottle Coffee as a 'Work Spot'. It's a great place for work meetings according to your notes.",
            sources=[
                {
                    "type": "location_visit",
                    "content": "Great place for work meetings",
                    "timestamp": datetime.utcnow() - timedelta(hours=2),
                    "relevance_score": 0.95,
                    "location": {
                        "address": "Blue Bottle Coffee, San Francisco, CA",
                        "names": [
                            "Blue Bottle Coffee",
                            "Coffee Shop",
                            "Work Spot",
                            "Meeting Place",
                        ],
                    },
                }
            ],
            media_references=[],
            session_id="location_search_123",
        )

        with patch(
            "app.services.query.query_processing_service.process_query",
            return_value=work_spot_query,
        ):
            work_response = await client.post(
                "/api/v1/query/",
                json={
                    "query": "Where are my work spots?",
                    "session_id": "location_search_123",
                },
                headers=auth_header,
            )

            assert work_response.status_code == status.HTTP_200_OK
            work_data = work_response.json()
            assert "Blue Bottle Coffee" in work_data["answer"]
            assert "Work Spot" in work_data["sources"][0]["location"]["names"]

        # Step 3: Search by category - "Food" related locations
        food_query = QueryResponse(
            answer="For food-related locations, you have the Ferry Building tagged as 'Food' and 'Weekend Spot' where you go for the Saturday farmers market.",
            sources=[
                {
                    "type": "location_visit",
                    "content": "Saturday farmers market",
                    "timestamp": datetime.utcnow() - timedelta(hours=2),
                    "relevance_score": 0.92,
                    "location": {
                        "address": "Ferry Building, San Francisco, CA",
                        "names": ["Ferry Building", "Market", "Food", "Weekend Spot"],
                    },
                }
            ],
            media_references=[],
            session_id="location_search_123",
        )

        with patch(
            "app.services.query.query_processing_service.process_query",
            return_value=food_query,
        ):
            food_response = await client.post(
                "/api/v1/query/",
                json={
                    "query": "Where do I go for food?",
                    "session_id": "location_search_123",
                },
                headers=auth_header,
            )

            assert food_response.status_code == status.HTTP_200_OK
            food_data = food_response.json()
            assert "Ferry Building" in food_data["answer"]
            assert "Food" in food_data["sources"][0]["location"]["names"]

        # Step 4: Update location tags
        updated_visit = DBLocationVisit(
            id=locations_with_tags[0]["id"],
            user_id=user_id,
            longitude=locations_with_tags[0]["longitude"],
            latitude=locations_with_tags[0]["latitude"],
            address=locations_with_tags[0]["address"],
            names=[
                "Golden Gate Park",
                "Park",
                "Nature",
                "Running Spot",
                "Photography Spot",
            ],
            visit_time=datetime.utcnow() - timedelta(hours=2),
            description="My favorite running location and great for photography",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        with patch(
            "app.repositories.location_visits.location_visit_repository.update_by_id",
            return_value=updated_visit,
        ):
            update_response = await client.put(
                f"/api/v1/locations/{locations_with_tags[0]['id']}",
                json={
                    "names": [
                        "Golden Gate Park",
                        "Park",
                        "Nature",
                        "Running Spot",
                        "Photography Spot",
                    ],
                    "description": "My favorite running location and great for photography",
                },
                headers=auth_header,
            )

            assert update_response.status_code == status.HTTP_200_OK
            update_data = update_response.json()
            assert "Photography Spot" in update_data["names"]
            assert "great for photography" in update_data["description"]

        # Step 5: Search with updated tags
        photography_query = QueryResponse(
            answer="You've tagged Golden Gate Park as a 'Photography Spot'. It's your favorite running location and great for photography according to your notes.",
            sources=[
                {
                    "type": "location_visit",
                    "content": "My favorite running location and great for photography",
                    "timestamp": datetime.utcnow() - timedelta(hours=2),
                    "relevance_score": 0.96,
                    "location": {
                        "address": "Golden Gate Park, San Francisco, CA",
                        "names": [
                            "Golden Gate Park",
                            "Park",
                            "Nature",
                            "Running Spot",
                            "Photography Spot",
                        ],
                    },
                }
            ],
            media_references=[],
            session_id="location_search_123",
        )

        with patch(
            "app.services.query.query_processing_service.process_query",
            return_value=photography_query,
        ):
            photo_response = await client.post(
                "/api/v1/query/",
                json={
                    "query": "Where are good photography spots?",
                    "session_id": "location_search_123",
                },
                headers=auth_header,
            )

            assert photo_response.status_code == status.HTTP_200_OK
            photo_data = photo_response.json()
            assert "Golden Gate Park" in photo_data["answer"]
            assert "Photography Spot" in photo_data["sources"][0]["location"]["names"]

    @pytest.mark.asyncio
    async def test_location_based_content_aggregation(
        self, client: AsyncClient, mock_db: AsyncSession, mock_auth_user: Dict[str, Any]
    ):
        """
        Test aggregating different types of content (notes, photos, voice) by location.
        """
        user_id = mock_auth_user["user"].id
        auth_header = {"Authorization": f"Bearer {mock_auth_user['token']}"}

        # Setup: Create content at the same location
        location_data = {
            "longitude": -122.4194,
            "latitude": 37.7749,
            "address": "Golden Gate Park, San Francisco, CA",
            "names": ["Golden Gate Park", "Park"],
        }

        # Create location visit
        visit_id = uuid4()
        db_visit = DBLocationVisit(
            id=visit_id,
            user_id=user_id,
            longitude=location_data["longitude"],
            latitude=location_data["latitude"],
            address=location_data["address"],
            names=location_data["names"],
            visit_time=datetime.utcnow() - timedelta(hours=3),
            duration=120,
            description="Beautiful day at the park",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        with patch(
            "app.repositories.location_visits.location_visit_repository.create",
            return_value=db_visit,
        ):
            await client.post(
                "/api/v1/locations",
                json={
                    "longitude": location_data["longitude"],
                    "latitude": location_data["latitude"],
                    "address": location_data["address"],
                    "names": location_data["names"],
                    "visit_time": (datetime.utcnow() - timedelta(hours=3)).isoformat(),
                    "duration": 120,
                    "description": "Beautiful day at the park",
                },
                headers=auth_header,
            )

        # Add text note at same location
        note_id = uuid4()
        db_note = DBTextNote(
            id=note_id,
            user_id=user_id,
            text_content="The cherry blossoms are absolutely stunning today. Perfect weather for a walk.",
            timestamp=datetime.utcnow() - timedelta(hours=2, minutes=30),
            longitude=location_data["longitude"],
            latitude=location_data["latitude"],
            address=location_data["address"],
            names=location_data["names"],
            created_at=datetime.utcnow(),
        )

        with (
            patch(
                "app.repositories.text_notes.text_note_repository.create",
                return_value=db_note,
            ),
            patch(
                "app.aws.embedding.embedding_service.process_and_store_text",
                return_value="doc_note_123",
            ),
            patch("app.services.usage.UsageService.increment_content_usage"),
        ):

            await client.post(
                "/api/v1/notes",
                json={
                    "text_content": "The cherry blossoms are absolutely stunning today. Perfect weather for a walk.",
                    "timestamp": (
                        datetime.utcnow() - timedelta(hours=2, minutes=30)
                    ).isoformat(),
                    "longitude": location_data["longitude"],
                    "latitude": location_data["latitude"],
                    "address": location_data["address"],
                    "names": location_data["names"],
                },
                headers=auth_header,
            )

        # Query for all content at this location
        location_aggregation_query = QueryResponse(
            answer="At Golden Gate Park today, you spent 2 hours enjoying a beautiful day. You noted that the cherry blossoms are absolutely stunning and the weather was perfect for a walk. You also took a photo of the blossoms and recorded a voice note about how peaceful the park was.",
            sources=[
                {
                    "type": "location_visit",
                    "content": "Beautiful day at the park",
                    "timestamp": datetime.utcnow() - timedelta(hours=3),
                    "relevance_score": 0.95,
                    "location": {"address": "Golden Gate Park, San Francisco, CA"},
                },
                {
                    "type": "text_note",
                    "content": "The cherry blossoms are absolutely stunning today. Perfect weather for a walk.",
                    "timestamp": datetime.utcnow() - timedelta(hours=2, minutes=30),
                    "relevance_score": 0.93,
                    "location": {"address": "Golden Gate Park, San Francisco, CA"},
                },
                {
                    "type": "image_desc",
                    "content": "Cherry blossoms in full bloom at Golden Gate Park",
                    "timestamp": datetime.utcnow() - timedelta(hours=2),
                    "relevance_score": 0.91,
                    "location": {"address": "Golden Gate Park, San Francisco, CA"},
                },
                {
                    "type": "voice_transcript",
                    "content": "This park is so peaceful today. The sound of birds and the gentle breeze through the trees is exactly what I needed.",
                    "timestamp": datetime.utcnow() - timedelta(hours=1, minutes=30),
                    "relevance_score": 0.89,
                    "location": {"address": "Golden Gate Park, San Francisco, CA"},
                },
            ],
            media_references=[
                MediaReference(
                    media_id=uuid4(),
                    media_type="photo",
                    description="Photo of cherry blossoms at Golden Gate Park",
                    timestamp=datetime.utcnow() - timedelta(hours=2),
                    location={"address": "Golden Gate Park, San Francisco, CA"},
                ),
                MediaReference(
                    media_id=uuid4(),
                    media_type="voice",
                    description="Voice note about the peaceful park atmosphere",
                    timestamp=datetime.utcnow() - timedelta(hours=1, minutes=30),
                    location={"address": "Golden Gate Park, San Francisco, CA"},
                ),
            ],
            session_id="location_aggregation_123",
        )

        with patch(
            "app.services.query.query_processing_service.process_query",
            return_value=location_aggregation_query,
        ):
            aggregation_response = await client.post(
                "/api/v1/query/",
                json={
                    "query": "Tell me everything about my visit to Golden Gate Park today",
                    "session_id": "location_aggregation_123",
                },
                headers=auth_header,
            )

            assert aggregation_response.status_code == status.HTTP_200_OK
            aggregation_data = aggregation_response.json()
            assert len(aggregation_data["sources"]) == 4  # Visit, note, photo, voice
            assert len(aggregation_data["media_references"]) == 2  # Photo and voice
            assert "cherry blossoms" in aggregation_data["answer"]
            assert "peaceful" in aggregation_data["answer"]
            assert "2 hours" in aggregation_data["answer"]

            # Verify all content types are represented
            source_types = [source["type"] for source in aggregation_data["sources"]]
            assert "location_visit" in source_types
            assert "text_note" in source_types
            assert "image_desc" in source_types
            assert "voice_transcript" in source_types
