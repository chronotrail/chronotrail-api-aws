"""
Tests for text notes repository operations.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import TextNoteCreate
from tests.conftest import TextNoteModel, UserModel
from tests.test_text_notes_repository_sqlite import test_text_note_repository


@pytest.fixture
async def test_user(async_db_session: AsyncSession):
    """Create a test user."""
    unique_id = str(uuid4())
    user = UserModel(
        email=f"test-{unique_id}@example.com",
        oauth_provider="google",
        oauth_subject=f"test_subject_{unique_id}",
        display_name="Test User",
    )
    async_db_session.add(user)
    await async_db_session.commit()
    await async_db_session.refresh(user)
    return user


@pytest.fixture
async def test_user_2(async_db_session: AsyncSession):
    """Create a second test user for isolation tests."""
    unique_id = str(uuid4())
    user = UserModel(
        email=f"test2-{unique_id}@example.com",
        oauth_provider="google",
        oauth_subject=f"test_subject_2_{unique_id}",
        display_name="Test User 2",
    )
    async_db_session.add(user)
    await async_db_session.commit()
    await async_db_session.refresh(user)
    return user


@pytest.fixture
def sample_text_note_data():
    """Sample text note creation data."""
    return TextNoteCreate(
        text_content="This is a test note about my visit to the coffee shop.",
        timestamp=datetime.utcnow() - timedelta(hours=1),
        longitude=Decimal("-122.4194"),
        latitude=Decimal("37.7749"),
        address="123 Main St, San Francisco, CA",
        names=["Coffee Shop", "Cafe"],
    )


@pytest.fixture
def sample_text_note_no_location():
    """Sample text note without location data."""
    return TextNoteCreate(
        text_content="This is a note without location data.",
        timestamp=datetime.utcnow() - timedelta(hours=2),
    )


class TestTextNoteRepository:
    """Test cases for text note repository operations."""

    async def test_create_text_note(
        self, async_db_session: AsyncSession, test_user, sample_text_note_data
    ):
        """Test creating a text note."""
        note = await test_text_note_repository.create(
            async_db_session, obj_in=sample_text_note_data, user_id=test_user.id
        )

        assert note.id is not None
        assert note.user_id == test_user.id
        assert note.text_content == sample_text_note_data.text_content
        assert note.timestamp == sample_text_note_data.timestamp
        assert note.longitude == sample_text_note_data.longitude
        assert note.latitude == sample_text_note_data.latitude
        assert note.address == sample_text_note_data.address
        assert note.names == sample_text_note_data.names
        assert note.created_at is not None

    async def test_create_text_note_no_location(
        self, async_db_session: AsyncSession, test_user, sample_text_note_no_location
    ):
        """Test creating a text note without location data."""
        note = await test_text_note_repository.create(
            async_db_session, obj_in=sample_text_note_no_location, user_id=test_user.id
        )

        assert note.id is not None
        assert note.user_id == test_user.id
        assert note.text_content == sample_text_note_no_location.text_content
        assert note.longitude is None
        assert note.latitude is None
        assert note.address is None
        assert note.names is None

    async def test_get_text_note(
        self, async_db_session: AsyncSession, test_user, sample_text_note_data
    ):
        """Test getting a text note by ID."""
        created_note = await test_text_note_repository.create(
            async_db_session, obj_in=sample_text_note_data, user_id=test_user.id
        )

        retrieved_note = await test_text_note_repository.get(
            async_db_session, created_note.id
        )

        assert retrieved_note is not None
        assert retrieved_note.id == created_note.id
        assert retrieved_note.text_content == sample_text_note_data.text_content

    async def test_get_text_note_not_found(self, async_db_session: AsyncSession):
        """Test getting a non-existent text note."""
        note = await test_text_note_repository.get(async_db_session, uuid4())
        assert note is None

    async def test_get_by_user(
        self,
        async_db_session: AsyncSession,
        test_user,
        test_user_2,
        sample_text_note_data,
    ):
        """Test getting a text note by user ID."""
        created_note = await test_text_note_repository.create(
            async_db_session, obj_in=sample_text_note_data, user_id=test_user.id
        )

        # Should find note for correct user
        retrieved_note = await test_text_note_repository.get_by_user(
            async_db_session, test_user.id, created_note.id
        )
        assert retrieved_note is not None
        assert retrieved_note.id == created_note.id

        # Should not find note for different user
        not_found = await test_text_note_repository.get_by_user(
            async_db_session, test_user_2.id, created_note.id
        )
        assert not_found is None

    async def test_get_by_user_with_date_range(
        self, async_db_session: AsyncSession, test_user
    ):
        """Test getting text notes by user with date range filtering."""
        now = datetime.utcnow()

        # Create notes at different times
        note1_data = TextNoteCreate(
            text_content="Note 1", timestamp=now - timedelta(days=3)
        )
        note2_data = TextNoteCreate(
            text_content="Note 2", timestamp=now - timedelta(days=1)
        )
        note3_data = TextNoteCreate(
            text_content="Note 3", timestamp=now - timedelta(hours=1)
        )

        await test_text_note_repository.create(
            async_db_session, obj_in=note1_data, user_id=test_user.id
        )
        await test_text_note_repository.create(
            async_db_session, obj_in=note2_data, user_id=test_user.id
        )
        await test_text_note_repository.create(
            async_db_session, obj_in=note3_data, user_id=test_user.id
        )

        # Test date range filtering
        start_date = now - timedelta(days=2)
        end_date = now

        notes = await test_text_note_repository.get_by_user_with_date_range(
            async_db_session, test_user.id, start_date, end_date
        )

        assert len(notes) == 2
        assert all(note.timestamp >= start_date for note in notes)
        assert all(note.timestamp <= end_date for note in notes)
        # Should be ordered by timestamp desc
        assert notes[0].timestamp > notes[1].timestamp

    async def test_get_by_location(self, async_db_session: AsyncSession, test_user):
        """Test getting text notes by location."""
        target_lng = Decimal("-122.4194")
        target_lat = Decimal("37.7749")

        # Create notes at different locations
        note1_data = TextNoteCreate(
            text_content="Near target",
            timestamp=datetime.utcnow() - timedelta(hours=1),
            longitude=target_lng,
            latitude=target_lat,
        )
        note2_data = TextNoteCreate(
            text_content="Also near target",
            timestamp=datetime.utcnow() - timedelta(hours=2),
            longitude=target_lng + Decimal("0.001"),  # Very close
            latitude=target_lat + Decimal("0.001"),
        )
        note3_data = TextNoteCreate(
            text_content="Far from target",
            timestamp=datetime.utcnow() - timedelta(hours=3),
            longitude=target_lng + Decimal("1.0"),  # Far away
            latitude=target_lat + Decimal("1.0"),
        )
        note4_data = TextNoteCreate(
            text_content="No location", timestamp=datetime.utcnow() - timedelta(hours=4)
        )

        await test_text_note_repository.create(
            async_db_session, obj_in=note1_data, user_id=test_user.id
        )
        await test_text_note_repository.create(
            async_db_session, obj_in=note2_data, user_id=test_user.id
        )
        await test_text_note_repository.create(
            async_db_session, obj_in=note3_data, user_id=test_user.id
        )
        await test_text_note_repository.create(
            async_db_session, obj_in=note4_data, user_id=test_user.id
        )

        # Search within 1km radius
        notes = await test_text_note_repository.get_by_location(
            async_db_session, test_user.id, target_lng, target_lat, radius_km=1.0
        )

        assert len(notes) == 2
        assert all(
            "Near target" in note.text_content
            or "Also near target" in note.text_content
            for note in notes
        )

    async def test_search_by_names(self, async_db_session: AsyncSession, test_user):
        """Test searching text notes by names/tags."""
        note1_data = TextNoteCreate(
            text_content="Coffee shop visit",
            timestamp=datetime.utcnow() - timedelta(hours=1),
            names=["Coffee Shop", "Starbucks"],
        )
        note2_data = TextNoteCreate(
            text_content="Restaurant dinner",
            timestamp=datetime.utcnow() - timedelta(hours=2),
            names=["Restaurant", "Italian"],
        )
        note3_data = TextNoteCreate(
            text_content="Park walk",
            timestamp=datetime.utcnow() - timedelta(hours=3),
            names=["Park", "Golden Gate"],
        )
        note4_data = TextNoteCreate(
            text_content="No names", timestamp=datetime.utcnow() - timedelta(hours=4)
        )

        await test_text_note_repository.create(
            async_db_session, obj_in=note1_data, user_id=test_user.id
        )
        await test_text_note_repository.create(
            async_db_session, obj_in=note2_data, user_id=test_user.id
        )
        await test_text_note_repository.create(
            async_db_session, obj_in=note3_data, user_id=test_user.id
        )
        await test_text_note_repository.create(
            async_db_session, obj_in=note4_data, user_id=test_user.id
        )

        # Search for coffee-related names
        notes = await test_text_note_repository.search_by_names(
            async_db_session, test_user.id, ["coffee", "starbucks"]
        )

        assert len(notes) == 1
        assert "Coffee shop visit" in notes[0].text_content

        # Search for park
        notes = await test_text_note_repository.search_by_names(
            async_db_session, test_user.id, ["park"]
        )

        assert len(notes) == 1
        assert "Park walk" in notes[0].text_content

    async def test_search_by_address(self, async_db_session: AsyncSession, test_user):
        """Test searching text notes by address."""
        note1_data = TextNoteCreate(
            text_content="Note 1",
            timestamp=datetime.utcnow() - timedelta(hours=1),
            address="123 Main St, San Francisco, CA",
        )
        note2_data = TextNoteCreate(
            text_content="Note 2",
            timestamp=datetime.utcnow() - timedelta(hours=2),
            address="456 Oak Ave, Oakland, CA",
        )
        note3_data = TextNoteCreate(
            text_content="Note 3", timestamp=datetime.utcnow() - timedelta(hours=3)
        )

        await test_text_note_repository.create(
            async_db_session, obj_in=note1_data, user_id=test_user.id
        )
        await test_text_note_repository.create(
            async_db_session, obj_in=note2_data, user_id=test_user.id
        )
        await test_text_note_repository.create(
            async_db_session, obj_in=note3_data, user_id=test_user.id
        )

        # Search for San Francisco
        notes = await test_text_note_repository.search_by_address(
            async_db_session, test_user.id, "San Francisco"
        )

        assert len(notes) == 1
        assert "Note 1" in notes[0].text_content

        # Search for CA (should find both with addresses)
        notes = await test_text_note_repository.search_by_address(
            async_db_session, test_user.id, "CA"
        )

        assert len(notes) == 2

    async def test_search_by_content(self, async_db_session: AsyncSession, test_user):
        """Test searching text notes by content."""
        note1_data = TextNoteCreate(
            text_content="I love coffee in the morning",
            timestamp=datetime.utcnow() - timedelta(hours=1),
        )
        note2_data = TextNoteCreate(
            text_content="Tea is better than coffee",
            timestamp=datetime.utcnow() - timedelta(hours=2),
        )
        note3_data = TextNoteCreate(
            text_content="Water is the best drink",
            timestamp=datetime.utcnow() - timedelta(hours=3),
        )

        await test_text_note_repository.create(
            async_db_session, obj_in=note1_data, user_id=test_user.id
        )
        await test_text_note_repository.create(
            async_db_session, obj_in=note2_data, user_id=test_user.id
        )
        await test_text_note_repository.create(
            async_db_session, obj_in=note3_data, user_id=test_user.id
        )

        # Search for coffee
        notes = await test_text_note_repository.search_by_content(
            async_db_session, test_user.id, "coffee"
        )

        assert len(notes) == 2
        assert all("coffee" in note.text_content.lower() for note in notes)

        # Search for water
        notes = await test_text_note_repository.search_by_content(
            async_db_session, test_user.id, "water"
        )

        assert len(notes) == 1
        assert "Water is the best drink" in notes[0].text_content

    async def test_get_with_location(self, async_db_session: AsyncSession, test_user):
        """Test getting text notes that have location data."""
        note1_data = TextNoteCreate(
            text_content="With location",
            timestamp=datetime.utcnow() - timedelta(hours=1),
            longitude=Decimal("-122.4194"),
            latitude=Decimal("37.7749"),
        )
        note2_data = TextNoteCreate(
            text_content="Without location",
            timestamp=datetime.utcnow() - timedelta(hours=2),
        )

        await test_text_note_repository.create(
            async_db_session, obj_in=note1_data, user_id=test_user.id
        )
        await test_text_note_repository.create(
            async_db_session, obj_in=note2_data, user_id=test_user.id
        )

        notes = await test_text_note_repository.get_with_location(
            async_db_session, test_user.id
        )

        assert len(notes) == 1
        assert "With location" in notes[0].text_content
        assert notes[0].longitude is not None
        assert notes[0].latitude is not None

    async def test_count_by_user(
        self, async_db_session: AsyncSession, test_user, test_user_2
    ):
        """Test counting text notes by user."""
        now = datetime.utcnow()

        # Create notes for user 1
        for i in range(3):
            note_data = TextNoteCreate(
                text_content=f"User 1 Note {i}", timestamp=now - timedelta(hours=i)
            )
            await test_text_note_repository.create(
                async_db_session, obj_in=note_data, user_id=test_user.id
            )

        # Create notes for user 2
        for i in range(2):
            note_data = TextNoteCreate(
                text_content=f"User 2 Note {i}", timestamp=now - timedelta(hours=i)
            )
            await test_text_note_repository.create(
                async_db_session, obj_in=note_data, user_id=test_user_2.id
            )

        # Count for user 1
        count1 = await test_text_note_repository.count_by_user(
            async_db_session, test_user.id
        )
        assert count1 == 3

        # Count for user 2
        count2 = await test_text_note_repository.count_by_user(
            async_db_session, test_user_2.id
        )
        assert count2 == 2

        # Count with date range
        start_date = now - timedelta(hours=1, minutes=30)
        count_recent = await test_text_note_repository.count_by_user(
            async_db_session, test_user.id, start_date=start_date
        )
        assert count_recent == 2  # Only notes from last 1.5 hours

    async def test_get_recent_by_user(self, async_db_session: AsyncSession, test_user):
        """Test getting recent text notes for a user."""
        now = datetime.utcnow()

        # Create notes at different times
        for i in range(5):
            note_data = TextNoteCreate(
                text_content=f"Note {i}", timestamp=now - timedelta(hours=i)
            )
            await test_text_note_repository.create(
                async_db_session, obj_in=note_data, user_id=test_user.id
            )

        # Get recent notes (limit 3)
        notes = await test_text_note_repository.get_recent_by_user(
            async_db_session, test_user.id, limit=3
        )

        assert len(notes) == 3
        # Should be ordered by timestamp desc (most recent first)
        assert notes[0].text_content == "Note 0"
        assert notes[1].text_content == "Note 1"
        assert notes[2].text_content == "Note 2"

    async def test_delete_by_user(
        self,
        async_db_session: AsyncSession,
        test_user,
        test_user_2,
        sample_text_note_data,
    ):
        """Test deleting a text note by user."""
        created_note = await test_text_note_repository.create(
            async_db_session, obj_in=sample_text_note_data, user_id=test_user.id
        )

        # Should be able to delete own note
        deleted_note = await test_text_note_repository.delete_by_user(
            async_db_session, test_user.id, created_note.id
        )
        assert deleted_note is not None
        assert deleted_note.id == created_note.id

        # Note should no longer exist
        retrieved_note = await test_text_note_repository.get(
            async_db_session, created_note.id
        )
        assert retrieved_note is None

        # Create another note
        another_note = await test_text_note_repository.create(
            async_db_session, obj_in=sample_text_note_data, user_id=test_user.id
        )

        # Different user should not be able to delete
        not_deleted = await test_text_note_repository.delete_by_user(
            async_db_session, test_user_2.id, another_note.id
        )
        assert not_deleted is None

        # Note should still exist
        still_exists = await test_text_note_repository.get(
            async_db_session, another_note.id
        )
        assert still_exists is not None

    async def test_user_isolation(
        self, async_db_session: AsyncSession, test_user, test_user_2
    ):
        """Test that users can only access their own text notes."""
        # Create notes for both users
        note1_data = TextNoteCreate(
            text_content="User 1 note", timestamp=datetime.utcnow() - timedelta(hours=1)
        )
        note2_data = TextNoteCreate(
            text_content="User 2 note", timestamp=datetime.utcnow() - timedelta(hours=2)
        )

        note1 = await test_text_note_repository.create(
            async_db_session, obj_in=note1_data, user_id=test_user.id
        )
        note2 = await test_text_note_repository.create(
            async_db_session, obj_in=note2_data, user_id=test_user_2.id
        )

        # User 1 should only see their own notes
        user1_notes = await test_text_note_repository.get_by_user_with_date_range(
            async_db_session, test_user.id
        )
        assert len(user1_notes) == 1
        assert user1_notes[0].text_content == "User 1 note"

        # User 2 should only see their own notes
        user2_notes = await test_text_note_repository.get_by_user_with_date_range(
            async_db_session, test_user_2.id
        )
        assert len(user2_notes) == 1
        assert user2_notes[0].text_content == "User 2 note"

        # Cross-user access should fail
        cross_access = await test_text_note_repository.get_by_user(
            async_db_session, test_user.id, note2.id
        )
        assert cross_access is None
