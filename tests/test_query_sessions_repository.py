"""
Tests for query sessions repository operations.
"""

import json
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import QuerySessionCreate, QuerySessionUpdate
from tests.conftest import UserModel
from tests.test_query_sessions_repository_sqlite import test_query_session_repository


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
def sample_session_context():
    """Sample session context data."""
    return {
        "conversation_history": [
            {"role": "user", "content": "What did I do yesterday?"},
            {
                "role": "assistant",
                "content": "Based on your data, you visited a coffee shop.",
            },
        ],
        "media_references": [
            {
                "media_id": str(uuid4()),
                "media_type": "photo",
                "description": "Photo from coffee shop",
                "referenced_at": datetime.utcnow().isoformat(),
            }
        ],
        "query_count": 1,
    }


class TestQuerySessionRepository:
    """Test cases for query session repository operations."""

    async def test_create_query_session(
        self, async_db_session: AsyncSession, test_user, sample_session_context
    ):
        """Test creating a query session."""
        session_data = QuerySessionCreate(
            user_id=test_user.id, session_context=sample_session_context
        )

        session = await test_query_session_repository.create(
            async_db_session, obj_in=session_data
        )

        assert session.id is not None
        assert session.user_id == test_user.id
        assert session.session_context == sample_session_context
        assert session.last_query is None
        assert session.created_at is not None
        assert session.updated_at is not None
        assert session.expires_at is not None
        assert session.expires_at > datetime.utcnow()

        # Should expire in approximately 1 hour
        expected_expiry = datetime.utcnow() + timedelta(hours=1)
        time_diff = abs((session.expires_at - expected_expiry).total_seconds())
        assert time_diff < 60  # Within 1 minute

    async def test_create_query_session_minimal(
        self, async_db_session: AsyncSession, test_user
    ):
        """Test creating a query session with minimal data."""
        session_data = QuerySessionCreate(user_id=test_user.id)

        session = await test_query_session_repository.create(
            async_db_session, obj_in=session_data
        )

        assert session.id is not None
        assert session.user_id == test_user.id
        assert session.session_context == {}
        assert session.last_query is None
        assert session.expires_at > datetime.utcnow()

    async def test_get_query_session(
        self, async_db_session: AsyncSession, test_user, sample_session_context
    ):
        """Test getting a query session by ID."""
        session_data = QuerySessionCreate(
            user_id=test_user.id, session_context=sample_session_context
        )

        created_session = await test_query_session_repository.create(
            async_db_session, obj_in=session_data
        )

        retrieved_session = await test_query_session_repository.get(
            async_db_session, created_session.id
        )

        assert retrieved_session is not None
        assert retrieved_session.id == created_session.id
        assert retrieved_session.session_context == sample_session_context

    async def test_get_query_session_not_found(self, async_db_session: AsyncSession):
        """Test getting a non-existent query session."""
        session = await test_query_session_repository.get(async_db_session, uuid4())
        assert session is None

    async def test_get_expired_session_returns_none(
        self, async_db_session: AsyncSession, test_user
    ):
        """Test that expired sessions are not returned."""
        session_data = QuerySessionCreate(user_id=test_user.id)

        created_session = await test_query_session_repository.create(
            async_db_session, obj_in=session_data
        )

        # Manually expire the session
        created_session.expires_at = datetime.utcnow() - timedelta(minutes=1)
        # Convert session_context back to JSON string for SQLite
        if isinstance(created_session.session_context, dict):
            created_session.session_context = json.dumps(
                created_session.session_context
            )
        async_db_session.add(created_session)
        await async_db_session.commit()

        # Should not be retrievable
        retrieved_session = await test_query_session_repository.get(
            async_db_session, created_session.id
        )
        assert retrieved_session is None

    async def test_get_by_user(
        self,
        async_db_session: AsyncSession,
        test_user,
        test_user_2,
        sample_session_context,
    ):
        """Test getting a query session by user ID."""
        session_data = QuerySessionCreate(
            user_id=test_user.id, session_context=sample_session_context
        )

        created_session = await test_query_session_repository.create(
            async_db_session, obj_in=session_data
        )

        # Should find session for correct user
        retrieved_session = await test_query_session_repository.get_by_user(
            async_db_session, test_user.id, created_session.id
        )
        assert retrieved_session is not None
        assert retrieved_session.id == created_session.id

        # Should not find session for different user
        not_found = await test_query_session_repository.get_by_user(
            async_db_session, test_user_2.id, created_session.id
        )
        assert not_found is None

    async def test_get_active_sessions_by_user(
        self, async_db_session: AsyncSession, test_user, test_user_2
    ):
        """Test getting active sessions for a user."""
        # Create sessions for user 1
        for i in range(3):
            session_data = QuerySessionCreate(
                user_id=test_user.id, session_context={"query_count": i}
            )
            await test_query_session_repository.create(
                async_db_session, obj_in=session_data
            )

        # Create sessions for user 2
        for i in range(2):
            session_data = QuerySessionCreate(
                user_id=test_user_2.id, session_context={"query_count": i}
            )
            await test_query_session_repository.create(
                async_db_session, obj_in=session_data
            )

        # Get sessions for user 1
        user1_sessions = (
            await test_query_session_repository.get_active_sessions_by_user(
                async_db_session, test_user.id
            )
        )
        assert len(user1_sessions) == 3
        assert all(session.user_id == test_user.id for session in user1_sessions)

        # Get sessions for user 2
        user2_sessions = (
            await test_query_session_repository.get_active_sessions_by_user(
                async_db_session, test_user_2.id
            )
        )
        assert len(user2_sessions) == 2
        assert all(session.user_id == test_user_2.id for session in user2_sessions)

    async def test_update_query_session(
        self, async_db_session: AsyncSession, test_user
    ):
        """Test updating a query session."""
        session_data = QuerySessionCreate(user_id=test_user.id)

        created_session = await test_query_session_repository.create(
            async_db_session, obj_in=session_data
        )

        original_expires_at = created_session.expires_at

        # Update session
        update_data = QuerySessionUpdate(
            session_context={"updated": True}, last_query="What's the weather like?"
        )

        updated_session = await test_query_session_repository.update(
            async_db_session, db_obj=created_session, obj_in=update_data
        )

        assert updated_session.session_context == {"updated": True}
        assert updated_session.last_query == "What's the weather like?"
        assert updated_session.updated_at > created_session.created_at
        assert updated_session.expires_at > original_expires_at  # Should be extended

    async def test_update_context(self, async_db_session: AsyncSession, test_user):
        """Test updating session context."""
        initial_context = {"query_count": 1}
        session_data = QuerySessionCreate(
            user_id=test_user.id, session_context=initial_context
        )

        created_session = await test_query_session_repository.create(
            async_db_session, obj_in=session_data
        )

        # Update context
        context_updates = {"query_count": 2, "new_field": "value"}
        last_query = "Tell me about my photos"

        updated_session = await test_query_session_repository.update_context(
            async_db_session,
            created_session.id,
            test_user.id,
            context_updates,
            last_query,
        )

        assert updated_session is not None
        assert updated_session.session_context["query_count"] == 2
        assert updated_session.session_context["new_field"] == "value"
        assert updated_session.last_query == last_query

    async def test_update_context_wrong_user(
        self, async_db_session: AsyncSession, test_user, test_user_2
    ):
        """Test updating context with wrong user ID."""
        session_data = QuerySessionCreate(user_id=test_user.id)

        created_session = await test_query_session_repository.create(
            async_db_session, obj_in=session_data
        )

        # Try to update with wrong user ID
        updated_session = await test_query_session_repository.update_context(
            async_db_session,
            created_session.id,
            test_user_2.id,  # Wrong user
            {"hacked": True},
        )

        assert updated_session is None

    async def test_add_media_reference(self, async_db_session: AsyncSession, test_user):
        """Test adding media references to session context."""
        session_data = QuerySessionCreate(user_id=test_user.id)

        created_session = await test_query_session_repository.create(
            async_db_session, obj_in=session_data
        )

        media_id = uuid4()

        # Add media reference
        updated_session = await test_query_session_repository.add_media_reference(
            async_db_session,
            created_session.id,
            test_user.id,
            media_id,
            "photo",
            "A beautiful sunset photo",
        )

        assert updated_session is not None
        media_refs = updated_session.session_context.get("media_references", [])
        assert len(media_refs) == 1
        assert media_refs[0]["media_id"] == str(media_id)
        assert media_refs[0]["media_type"] == "photo"
        assert media_refs[0]["description"] == "A beautiful sunset photo"
        assert "referenced_at" in media_refs[0]

        # Add another media reference
        media_id_2 = uuid4()
        updated_session = await test_query_session_repository.add_media_reference(
            async_db_session,
            created_session.id,
            test_user.id,
            media_id_2,
            "voice",
            "Voice note from meeting",
        )

        media_refs = updated_session.session_context.get("media_references", [])
        assert len(media_refs) == 2

    async def test_get_media_references(
        self, async_db_session: AsyncSession, test_user
    ):
        """Test getting media references from session."""
        media_refs = [
            {
                "media_id": str(uuid4()),
                "media_type": "photo",
                "description": "Photo 1",
                "referenced_at": datetime.utcnow().isoformat(),
            },
            {
                "media_id": str(uuid4()),
                "media_type": "voice",
                "description": "Voice note 1",
                "referenced_at": datetime.utcnow().isoformat(),
            },
        ]

        session_data = QuerySessionCreate(
            user_id=test_user.id, session_context={"media_references": media_refs}
        )

        created_session = await test_query_session_repository.create(
            async_db_session, obj_in=session_data
        )

        # Get media references
        retrieved_refs = await test_query_session_repository.get_media_references(
            async_db_session, created_session.id, test_user.id
        )

        assert len(retrieved_refs) == 2
        assert retrieved_refs == media_refs

    async def test_get_media_references_empty(
        self, async_db_session: AsyncSession, test_user
    ):
        """Test getting media references from session with no references."""
        session_data = QuerySessionCreate(user_id=test_user.id)

        created_session = await test_query_session_repository.create(
            async_db_session, obj_in=session_data
        )

        # Get media references
        retrieved_refs = await test_query_session_repository.get_media_references(
            async_db_session, created_session.id, test_user.id
        )

        assert retrieved_refs == []

    async def test_extend_expiration(self, async_db_session: AsyncSession, test_user):
        """Test extending session expiration."""
        session_data = QuerySessionCreate(user_id=test_user.id)

        created_session = await test_query_session_repository.create(
            async_db_session, obj_in=session_data
        )

        original_expiry = created_session.expires_at

        # Extend expiration by 2 hours
        extended_session = await test_query_session_repository.extend_expiration(
            async_db_session, created_session.id, test_user.id, hours=2
        )

        assert extended_session is not None
        assert extended_session.expires_at > original_expiry

        # Should be extended by approximately 2 hours from now
        expected_expiry = datetime.utcnow() + timedelta(hours=2)
        time_diff = abs((extended_session.expires_at - expected_expiry).total_seconds())
        assert time_diff < 60  # Within 1 minute

    async def test_cleanup_expired_sessions(
        self, async_db_session: AsyncSession, test_user
    ):
        """Test cleaning up expired sessions."""
        # Create active sessions
        for i in range(3):
            session_data = QuerySessionCreate(user_id=test_user.id)
            await test_query_session_repository.create(
                async_db_session, obj_in=session_data
            )

        # Create expired sessions
        for i in range(2):
            session_data = QuerySessionCreate(user_id=test_user.id)
            expired_session = await test_query_session_repository.create(
                async_db_session, obj_in=session_data
            )

            # Manually expire the session
            expired_session.expires_at = datetime.utcnow() - timedelta(minutes=1)
            # Convert session_context back to JSON string for SQLite
            if isinstance(expired_session.session_context, dict):
                expired_session.session_context = json.dumps(
                    expired_session.session_context
                )
            async_db_session.add(expired_session)
            await async_db_session.commit()

        # Cleanup expired sessions
        deleted_count = await test_query_session_repository.cleanup_expired_sessions(
            async_db_session
        )

        assert deleted_count == 2

        # Verify active sessions still exist
        active_sessions = (
            await test_query_session_repository.get_active_sessions_by_user(
                async_db_session, test_user.id
            )
        )
        assert len(active_sessions) == 3

    async def test_cleanup_old_sessions_by_user(
        self, async_db_session: AsyncSession, test_user
    ):
        """Test cleaning up old sessions for a user."""
        # Create 5 sessions with different update times
        sessions = []
        for i in range(5):
            session_data = QuerySessionCreate(user_id=test_user.id)
            session = await test_query_session_repository.create(
                async_db_session, obj_in=session_data
            )

            # Set different update times
            session.updated_at = datetime.utcnow() - timedelta(hours=i)
            # Convert session_context back to JSON string for SQLite
            if isinstance(session.session_context, dict):
                session.session_context = json.dumps(session.session_context)
            async_db_session.add(session)
            sessions.append(session)

        await async_db_session.commit()

        # Keep only 3 most recent sessions
        deleted_count = (
            await test_query_session_repository.cleanup_old_sessions_by_user(
                async_db_session, test_user.id, keep_count=3
            )
        )

        assert deleted_count == 2

        # Verify only 3 sessions remain
        remaining_sessions = (
            await test_query_session_repository.get_active_sessions_by_user(
                async_db_session, test_user.id
            )
        )
        assert len(remaining_sessions) == 3

    async def test_count_active_by_user(
        self, async_db_session: AsyncSession, test_user, test_user_2
    ):
        """Test counting active sessions by user."""
        # Create sessions for user 1
        for i in range(3):
            session_data = QuerySessionCreate(user_id=test_user.id)
            await test_query_session_repository.create(
                async_db_session, obj_in=session_data
            )

        # Create sessions for user 2
        for i in range(2):
            session_data = QuerySessionCreate(user_id=test_user_2.id)
            await test_query_session_repository.create(
                async_db_session, obj_in=session_data
            )

        # Create expired session for user 1
        expired_session_data = QuerySessionCreate(user_id=test_user.id)
        expired_session = await test_query_session_repository.create(
            async_db_session, obj_in=expired_session_data
        )
        expired_session.expires_at = datetime.utcnow() - timedelta(minutes=1)
        # Convert session_context back to JSON string for SQLite
        if isinstance(expired_session.session_context, dict):
            expired_session.session_context = json.dumps(
                expired_session.session_context
            )
        async_db_session.add(expired_session)
        await async_db_session.commit()

        # Count active sessions
        count1 = await test_query_session_repository.count_active_by_user(
            async_db_session, test_user.id
        )
        assert count1 == 3  # Expired session not counted

        count2 = await test_query_session_repository.count_active_by_user(
            async_db_session, test_user_2.id
        )
        assert count2 == 2

    async def test_get_session_statistics(
        self, async_db_session: AsyncSession, test_user
    ):
        """Test getting session statistics."""
        # Create active sessions with different durations
        for i in range(3):
            session_data = QuerySessionCreate(user_id=test_user.id)
            session = await test_query_session_repository.create(
                async_db_session, obj_in=session_data
            )

            # Simulate different session durations
            session.created_at = datetime.utcnow() - timedelta(minutes=30)
            session.updated_at = datetime.utcnow() - timedelta(minutes=30 - (i * 10))
            # Convert session_context back to JSON string for SQLite
            if isinstance(session.session_context, dict):
                session.session_context = json.dumps(session.session_context)
            async_db_session.add(session)

        # Create expired session
        expired_session_data = QuerySessionCreate(user_id=test_user.id)
        expired_session = await test_query_session_repository.create(
            async_db_session, obj_in=expired_session_data
        )
        expired_session.expires_at = datetime.utcnow() - timedelta(minutes=1)
        # Convert session_context back to JSON string for SQLite
        if isinstance(expired_session.session_context, dict):
            expired_session.session_context = json.dumps(
                expired_session.session_context
            )
        async_db_session.add(expired_session)

        await async_db_session.commit()

        # Get statistics for user
        stats = await test_query_session_repository.get_session_statistics(
            async_db_session, test_user.id
        )

        assert stats["active_sessions"] == 3
        assert stats["expired_sessions"] == 1
        assert stats["total_sessions"] == 4
        assert stats["average_duration_minutes"] >= 0

    async def test_delete_by_user(
        self, async_db_session: AsyncSession, test_user, test_user_2
    ):
        """Test deleting a session by user."""
        session_data = QuerySessionCreate(user_id=test_user.id)

        created_session = await test_query_session_repository.create(
            async_db_session, obj_in=session_data
        )

        # Should be able to delete own session
        deleted_session = await test_query_session_repository.delete_by_user(
            async_db_session, test_user.id, created_session.id
        )
        assert deleted_session is not None
        assert deleted_session.id == created_session.id

        # Session should no longer exist
        retrieved_session = await test_query_session_repository.get(
            async_db_session, created_session.id
        )
        assert retrieved_session is None

        # Create another session
        another_session_data = QuerySessionCreate(user_id=test_user.id)
        another_session = await test_query_session_repository.create(
            async_db_session, obj_in=another_session_data
        )

        # Different user should not be able to delete
        not_deleted = await test_query_session_repository.delete_by_user(
            async_db_session, test_user_2.id, another_session.id
        )
        assert not_deleted is None

        # Session should still exist
        still_exists = await test_query_session_repository.get(
            async_db_session, another_session.id
        )
        assert still_exists is not None

    async def test_user_isolation(
        self, async_db_session: AsyncSession, test_user, test_user_2
    ):
        """Test that users can only access their own sessions."""
        # Create sessions for both users
        session1_data = QuerySessionCreate(
            user_id=test_user.id, session_context={"user": "user1"}
        )
        session2_data = QuerySessionCreate(
            user_id=test_user_2.id, session_context={"user": "user2"}
        )

        session1 = await test_query_session_repository.create(
            async_db_session, obj_in=session1_data
        )
        session2 = await test_query_session_repository.create(
            async_db_session, obj_in=session2_data
        )

        # User 1 should only see their own sessions
        user1_sessions = (
            await test_query_session_repository.get_active_sessions_by_user(
                async_db_session, test_user.id
            )
        )
        assert len(user1_sessions) == 1
        assert user1_sessions[0].session_context["user"] == "user1"

        # User 2 should only see their own sessions
        user2_sessions = (
            await test_query_session_repository.get_active_sessions_by_user(
                async_db_session, test_user_2.id
            )
        )
        assert len(user2_sessions) == 1
        assert user2_sessions[0].session_context["user"] == "user2"

        # Cross-user access should fail
        cross_access = await test_query_session_repository.get_by_user(
            async_db_session, test_user.id, session2.id
        )
        assert cross_access is None

    async def test_filters_in_get_multi(
        self, async_db_session: AsyncSession, test_user
    ):
        """Test filtering in get_multi method."""
        # Create sessions with different properties
        session1_data = QuerySessionCreate(
            user_id=test_user.id,
            session_context={"media_references": [{"media_id": "123"}]},
        )
        session2_data = QuerySessionCreate(user_id=test_user.id)

        session1 = await test_query_session_repository.create(
            async_db_session, obj_in=session1_data
        )
        session2 = await test_query_session_repository.create(
            async_db_session, obj_in=session2_data
        )

        # Update one session with a query
        await test_query_session_repository.update_context(
            async_db_session, session1.id, test_user.id, {}, "test query"
        )

        # Test filtering by user
        user_sessions = await test_query_session_repository.get_multi(
            async_db_session, filters={"user_id": test_user.id}
        )
        assert len(user_sessions) == 2

        # Test filtering by media references
        media_sessions = await test_query_session_repository.get_multi(
            async_db_session, filters={"has_media_references": True}
        )
        assert len(media_sessions) == 1

        # Test filtering by query content
        query_sessions = await test_query_session_repository.get_multi(
            async_db_session, filters={"last_query_contains": "test"}
        )
        assert len(query_sessions) == 1
