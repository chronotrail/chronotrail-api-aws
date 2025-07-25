"""
Repository for query sessions with CRUD operations, session management,
and cleanup functionality.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import QuerySession as DBQuerySession
from app.models.schemas import QuerySessionCreate, QuerySessionUpdate
from app.repositories.base import BaseRepository


class QuerySessionRepository(
    BaseRepository[DBQuerySession, QuerySessionCreate, QuerySessionUpdate]
):
    """
    Repository for query session operations with conversation context management,
    session expiration handling, and media reference tracking.
    """

    def __init__(self):
        super().__init__(DBQuerySession)

    async def create(
        self, db: AsyncSession, *, obj_in: QuerySessionCreate, **kwargs
    ) -> DBQuerySession:
        """
        Create a new query session.

        Args:
            db: Database session
            obj_in: Query session creation data

        Returns:
            Created query session
        """
        # Set default expiration to 1 hour from now
        expires_at = datetime.utcnow() + timedelta(hours=1)

        db_obj = DBQuerySession(
            user_id=obj_in.user_id,
            session_context=obj_in.session_context or {},
            expires_at=expires_at,
        )

        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)

        return db_obj

    async def get(self, db: AsyncSession, id: UUID) -> Optional[DBQuerySession]:
        """
        Get a query session by ID.

        Args:
            db: Database session
            id: Query session ID

        Returns:
            Query session if found and not expired, None otherwise
        """
        stmt = select(DBQuerySession).where(
            and_(DBQuerySession.id == id, DBQuerySession.expires_at > datetime.utcnow())
        )
        result = await db.execute(stmt)
        return result.scalars().first()

    async def get_by_user(
        self, db: AsyncSession, user_id: UUID, session_id: UUID
    ) -> Optional[DBQuerySession]:
        """
        Get a query session by ID for a specific user.

        Args:
            db: Database session
            user_id: User ID
            session_id: Query session ID

        Returns:
            Query session if found, belongs to user, and not expired, None otherwise
        """
        stmt = select(DBQuerySession).where(
            and_(
                DBQuerySession.id == session_id,
                DBQuerySession.user_id == user_id,
                DBQuerySession.expires_at > datetime.utcnow(),
            )
        )
        result = await db.execute(stmt)
        return result.scalars().first()

    async def get_multi(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DBQuerySession]:
        """
        Get multiple query sessions with optional filtering.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters to apply

        Returns:
            List of query sessions
        """
        stmt = select(DBQuerySession).where(
            DBQuerySession.expires_at > datetime.utcnow()
        )

        # Apply filters if provided
        if filters:
            stmt = self._apply_filters(stmt, filters)

        stmt = stmt.offset(skip).limit(limit).order_by(DBQuerySession.updated_at.desc())

        result = await db.execute(stmt)
        return result.scalars().all()

    async def get_active_sessions_by_user(
        self, db: AsyncSession, user_id: UUID, skip: int = 0, limit: int = 10
    ) -> List[DBQuerySession]:
        """
        Get active query sessions for a user.

        Args:
            db: Database session
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of active query sessions for the user
        """
        stmt = (
            select(DBQuerySession)
            .where(
                and_(
                    DBQuerySession.user_id == user_id,
                    DBQuerySession.expires_at > datetime.utcnow(),
                )
            )
            .order_by(DBQuerySession.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )

        result = await db.execute(stmt)
        return result.scalars().all()

    async def update(
        self, db: AsyncSession, *, db_obj: DBQuerySession, obj_in: QuerySessionUpdate
    ) -> DBQuerySession:
        """
        Update a query session with new context or query information.

        Args:
            db: Database session
            db_obj: Existing query session object
            obj_in: Update data

        Returns:
            Updated query session
        """
        update_data = obj_in.model_dump(exclude_unset=True)

        # Always update the updated_at timestamp
        update_data["updated_at"] = datetime.utcnow()

        # Extend session expiration by 1 hour when updated
        update_data["expires_at"] = datetime.utcnow() + timedelta(hours=1)

        for field, value in update_data.items():
            setattr(db_obj, field, value)

        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)

        return db_obj

    async def update_context(
        self,
        db: AsyncSession,
        session_id: UUID,
        user_id: UUID,
        context_updates: Dict[str, Any],
        last_query: Optional[str] = None,
    ) -> Optional[DBQuerySession]:
        """
        Update session context with new conversation data.

        Args:
            db: Database session
            session_id: Query session ID
            user_id: User ID (for security)
            context_updates: Context data to merge
            last_query: Latest query string

        Returns:
            Updated query session if found and belongs to user, None otherwise
        """
        db_obj = await self.get_by_user(db, user_id, session_id)
        if not db_obj:
            return None

        # Merge context updates with existing context
        current_context = db_obj.session_context or {}
        current_context.update(context_updates)

        update_data = QuerySessionUpdate(
            session_context=current_context, last_query=last_query
        )

        return await self.update(db, db_obj=db_obj, obj_in=update_data)

    async def add_media_reference(
        self,
        db: AsyncSession,
        session_id: UUID,
        user_id: UUID,
        media_id: UUID,
        media_type: str,
        description: str,
    ) -> Optional[DBQuerySession]:
        """
        Add a media reference to the session context.

        Args:
            db: Database session
            session_id: Query session ID
            user_id: User ID (for security)
            media_id: ID of the referenced media
            media_type: Type of media ('photo', 'voice')
            description: Description of the media reference

        Returns:
            Updated query session if found and belongs to user, None otherwise
        """
        db_obj = await self.get_by_user(db, user_id, session_id)
        if not db_obj:
            return None

        # Get current context and ensure media_references exists
        current_context = db_obj.session_context or {}
        media_references = current_context.get("media_references", [])

        # Add new media reference
        media_ref = {
            "media_id": str(media_id),
            "media_type": media_type,
            "description": description,
            "referenced_at": datetime.utcnow().isoformat(),
        }

        media_references.append(media_ref)
        current_context["media_references"] = media_references

        update_data = QuerySessionUpdate(session_context=current_context)
        return await self.update(db, db_obj=db_obj, obj_in=update_data)

    async def get_media_references(
        self, db: AsyncSession, session_id: UUID, user_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Get all media references from a session.

        Args:
            db: Database session
            session_id: Query session ID
            user_id: User ID (for security)

        Returns:
            List of media references from the session
        """
        db_obj = await self.get_by_user(db, user_id, session_id)
        if not db_obj or not db_obj.session_context:
            return []

        return db_obj.session_context.get("media_references", [])

    async def extend_expiration(
        self, db: AsyncSession, session_id: UUID, user_id: UUID, hours: int = 1
    ) -> Optional[DBQuerySession]:
        """
        Extend the expiration time of a session.

        Args:
            db: Database session
            session_id: Query session ID
            user_id: User ID (for security)
            hours: Number of hours to extend

        Returns:
            Updated query session if found and belongs to user, None otherwise
        """
        db_obj = await self.get_by_user(db, user_id, session_id)
        if not db_obj:
            return None

        # Extend expiration
        new_expiration = datetime.utcnow() + timedelta(hours=hours)
        db_obj.expires_at = new_expiration
        db_obj.updated_at = datetime.utcnow()

        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)

        return db_obj

    async def delete(self, db: AsyncSession, *, id: UUID) -> Optional[DBQuerySession]:
        """
        Delete a query session by ID.

        Args:
            db: Database session
            id: Query session ID

        Returns:
            Deleted query session if found, None otherwise
        """
        db_obj = await self.get(db, id)
        if db_obj:
            await db.delete(db_obj)
            await db.commit()
        return db_obj

    async def delete_by_user(
        self, db: AsyncSession, user_id: UUID, session_id: UUID
    ) -> Optional[DBQuerySession]:
        """
        Delete a query session by ID for a specific user.

        Args:
            db: Database session
            user_id: User ID
            session_id: Query session ID

        Returns:
            Deleted query session if found and belongs to user, None otherwise
        """
        db_obj = await self.get_by_user(db, user_id, session_id)
        if db_obj:
            await db.delete(db_obj)
            await db.commit()
        return db_obj

    async def cleanup_expired_sessions(self, db: AsyncSession) -> int:
        """
        Delete all expired query sessions.

        Args:
            db: Database session

        Returns:
            Number of sessions deleted
        """
        # Get count of expired sessions first
        count_stmt = select(func.count(DBQuerySession.id)).where(
            DBQuerySession.expires_at <= datetime.utcnow()
        )
        result = await db.execute(count_stmt)
        expired_count = result.scalar()

        # Delete expired sessions
        delete_stmt = delete(DBQuerySession).where(
            DBQuerySession.expires_at <= datetime.utcnow()
        )
        await db.execute(delete_stmt)
        await db.commit()

        return expired_count

    async def cleanup_old_sessions_by_user(
        self, db: AsyncSession, user_id: UUID, keep_count: int = 10
    ) -> int:
        """
        Clean up old sessions for a user, keeping only the most recent ones.

        Args:
            db: Database session
            user_id: User ID
            keep_count: Number of recent sessions to keep

        Returns:
            Number of sessions deleted
        """
        # Get sessions to keep (most recent ones)
        keep_stmt = (
            select(DBQuerySession.id)
            .where(DBQuerySession.user_id == user_id)
            .order_by(DBQuerySession.updated_at.desc())
            .limit(keep_count)
        )

        keep_result = await db.execute(keep_stmt)
        keep_ids = [row[0] for row in keep_result.fetchall()]

        if not keep_ids:
            return 0

        # Count sessions to delete
        count_stmt = select(func.count(DBQuerySession.id)).where(
            and_(DBQuerySession.user_id == user_id, ~DBQuerySession.id.in_(keep_ids))
        )
        result = await db.execute(count_stmt)
        delete_count = result.scalar()

        # Delete old sessions
        delete_stmt = delete(DBQuerySession).where(
            and_(DBQuerySession.user_id == user_id, ~DBQuerySession.id.in_(keep_ids))
        )
        await db.execute(delete_stmt)
        await db.commit()

        return delete_count

    async def count(
        self, db: AsyncSession, filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count query sessions with optional filtering.

        Args:
            db: Database session
            filters: Optional filters to apply

        Returns:
            Count of matching query sessions
        """
        stmt = select(func.count(DBQuerySession.id)).where(
            DBQuerySession.expires_at > datetime.utcnow()
        )

        if filters:
            stmt = self._apply_filters(stmt, filters)

        result = await db.execute(stmt)
        return result.scalar()

    async def count_active_by_user(self, db: AsyncSession, user_id: UUID) -> int:
        """
        Count active query sessions for a user.

        Args:
            db: Database session
            user_id: User ID

        Returns:
            Count of active sessions for the user
        """
        stmt = select(func.count(DBQuerySession.id)).where(
            and_(
                DBQuerySession.user_id == user_id,
                DBQuerySession.expires_at > datetime.utcnow(),
            )
        )

        result = await db.execute(stmt)
        return result.scalar()

    async def get_session_statistics(
        self, db: AsyncSession, user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about query sessions.

        Args:
            db: Database session
            user_id: Optional user ID to filter by

        Returns:
            Dictionary with session statistics
        """
        base_filter = DBQuerySession.expires_at > datetime.utcnow()
        if user_id:
            base_filter = and_(base_filter, DBQuerySession.user_id == user_id)

        # Count active sessions
        active_stmt = select(func.count(DBQuerySession.id)).where(base_filter)
        active_result = await db.execute(active_stmt)
        active_count = active_result.scalar()

        # Count expired sessions
        expired_filter = DBQuerySession.expires_at <= datetime.utcnow()
        if user_id:
            expired_filter = and_(expired_filter, DBQuerySession.user_id == user_id)

        expired_stmt = select(func.count(DBQuerySession.id)).where(expired_filter)
        expired_result = await db.execute(expired_stmt)
        expired_count = expired_result.scalar()

        # Get average session duration (for active sessions)
        duration_stmt = select(
            func.avg(
                func.extract(
                    "epoch", DBQuerySession.updated_at - DBQuerySession.created_at
                )
            )
        ).where(base_filter)
        duration_result = await db.execute(duration_stmt)
        avg_duration_seconds = duration_result.scalar() or 0

        return {
            "active_sessions": active_count,
            "expired_sessions": expired_count,
            "total_sessions": active_count + expired_count,
            "average_duration_minutes": (
                round(avg_duration_seconds / 60, 2) if avg_duration_seconds else 0
            ),
        }

    def _apply_filters(self, stmt, filters: Dict[str, Any]):
        """
        Apply filters to a SQLAlchemy statement.

        Args:
            stmt: SQLAlchemy statement
            filters: Dictionary of filters to apply

        Returns:
            Modified statement with filters applied
        """
        if "user_id" in filters:
            stmt = stmt.where(DBQuerySession.user_id == filters["user_id"])

        if "created_after" in filters:
            stmt = stmt.where(DBQuerySession.created_at >= filters["created_after"])

        if "created_before" in filters:
            stmt = stmt.where(DBQuerySession.created_at <= filters["created_before"])

        if "updated_after" in filters:
            stmt = stmt.where(DBQuerySession.updated_at >= filters["updated_after"])

        if "updated_before" in filters:
            stmt = stmt.where(DBQuerySession.updated_at <= filters["updated_before"])

        if "has_media_references" in filters:
            if filters["has_media_references"]:
                stmt = stmt.where(
                    and_(
                        DBQuerySession.session_context.isnot(None),
                        DBQuerySession.session_context.op("->>")(
                            "media_references"
                        ).isnot(None),
                    )
                )
            else:
                stmt = stmt.where(
                    or_(
                        DBQuerySession.session_context.is_(None),
                        DBQuerySession.session_context.op("->>")(
                            "media_references"
                        ).is_(None),
                    )
                )

        if "last_query_contains" in filters:
            stmt = stmt.where(
                and_(
                    DBQuerySession.last_query.isnot(None),
                    DBQuerySession.last_query.ilike(
                        f"%{filters['last_query_contains']}%"
                    ),
                )
            )

        return stmt


# Create a singleton instance
query_session_repository = QuerySessionRepository()
