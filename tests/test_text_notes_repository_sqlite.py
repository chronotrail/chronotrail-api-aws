"""
SQLite-compatible repository for testing text notes.
"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import TextNoteCreate
from tests.conftest import TextNoteModel, UserModel


class TextNoteModelRepository:
    """
    Test repository for text note operations with SQLite compatibility.
    """

    def __init__(self):
        self.model = TextNoteModel

    async def create(
        self, db: AsyncSession, *, obj_in: TextNoteCreate, user_id: UUID
    ) -> TextNoteModel:
        """Create a new text note."""
        # Convert names list to JSON string for SQLite
        names_json = json.dumps(obj_in.names) if obj_in.names else None

        db_obj = TextNoteModel(
            user_id=str(user_id),
            text_content=obj_in.text_content,
            timestamp=obj_in.timestamp,
            longitude=obj_in.longitude,
            latitude=obj_in.latitude,
            address=obj_in.address,
            names=names_json,
        )

        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)

        # Detach from session to prevent further tracking
        db.expunge(db_obj)

        # Convert names back to list for response
        if db_obj.names:
            try:
                db_obj.names = json.loads(db_obj.names)
            except (json.JSONDecodeError, TypeError):
                db_obj.names = []

        return db_obj

    async def get(self, db: AsyncSession, id: UUID) -> Optional[TextNoteModel]:
        """Get a text note by ID."""
        stmt = select(TextNoteModel).where(TextNoteModel.id == str(id))
        result = await db.execute(stmt)
        note = result.scalars().first()

        if note and note.names:
            try:
                note.names = json.loads(note.names)
            except (json.JSONDecodeError, TypeError):
                note.names = []

        return note

    async def get_by_user(
        self, db: AsyncSession, user_id: UUID, note_id: UUID
    ) -> Optional[TextNoteModel]:
        """Get a text note by ID for a specific user."""
        with db.no_autoflush:
            stmt = select(TextNoteModel).where(
                and_(
                    TextNoteModel.id == str(note_id),
                    TextNoteModel.user_id == str(user_id),
                )
            )
            result = await db.execute(stmt)
            note = result.scalars().first()

            if note and note.names:
                try:
                    note.names = json.loads(note.names)
                except (json.JSONDecodeError, TypeError):
                    note.names = []

        return note

    async def get_by_user_with_date_range(
        self,
        db: AsyncSession,
        user_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[TextNoteModel]:
        """Get text notes for a user within a date range."""
        stmt = select(TextNoteModel).where(TextNoteModel.user_id == str(user_id))

        if start_date:
            stmt = stmt.where(TextNoteModel.timestamp >= start_date)
        if end_date:
            stmt = stmt.where(TextNoteModel.timestamp <= end_date)

        stmt = stmt.order_by(TextNoteModel.timestamp.desc()).offset(skip).limit(limit)

        result = await db.execute(stmt)
        notes = result.scalars().all()

        # Convert names back to lists
        for note in notes:
            if note.names:
                try:
                    note.names = json.loads(note.names)
                except (json.JSONDecodeError, TypeError):
                    note.names = []

        return notes

    async def search_by_names(
        self,
        db: AsyncSession,
        user_id: UUID,
        search_terms: List[str],
        skip: int = 0,
        limit: int = 100,
    ) -> List[TextNoteModel]:
        """Search text notes by names/tags."""
        with db.no_autoflush:
            stmt = select(TextNoteModel).where(TextNoteModel.user_id == str(user_id))

            if search_terms:
                # For SQLite, search within the JSON string
                conditions = []
                for term in search_terms:
                    conditions.append(TextNoteModel.names.like(f'%"{term}"%'))

                if conditions:
                    stmt = stmt.where(or_(*conditions))

            stmt = (
                stmt.order_by(TextNoteModel.timestamp.desc()).offset(skip).limit(limit)
            )

            result = await db.execute(stmt)
            notes = result.scalars().all()

            # Convert names back to lists
            for note in notes:
                if note.names:
                    try:
                        note.names = json.loads(note.names)
                    except (json.JSONDecodeError, TypeError):
                        note.names = []

        return notes

    async def search_by_content(
        self,
        db: AsyncSession,
        user_id: UUID,
        content_query: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[TextNoteModel]:
        """Search text notes by content."""
        stmt = (
            select(TextNoteModel)
            .where(
                and_(
                    TextNoteModel.user_id == str(user_id),
                    TextNoteModel.text_content.like(f"%{content_query}%"),
                )
            )
            .order_by(TextNoteModel.timestamp.desc())
            .offset(skip)
            .limit(limit)
        )

        result = await db.execute(stmt)
        notes = result.scalars().all()

        # Convert names back to lists
        for note in notes:
            if note.names:
                try:
                    note.names = json.loads(note.names)
                except (json.JSONDecodeError, TypeError):
                    note.names = []

        return notes

    async def delete_by_user(
        self, db: AsyncSession, user_id: UUID, note_id: UUID
    ) -> Optional[TextNoteModel]:
        """Delete a text note by ID for a specific user."""
        # Use no_autoflush to prevent premature flushing of pending changes
        with db.no_autoflush:
            db_obj = await self.get_by_user(db, user_id, note_id)
            if db_obj:
                stmt = delete(TextNoteModel).where(
                    and_(
                        TextNoteModel.id == str(note_id),
                        TextNoteModel.user_id == str(user_id),
                    )
                )
                await db.execute(stmt)
                await db.commit()
        return db_obj

    async def count_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Count text notes for a user within a date range."""
        stmt = select(func.count(TextNoteModel.id)).where(
            TextNoteModel.user_id == str(user_id)
        )

        if start_date:
            stmt = stmt.where(TextNoteModel.timestamp >= start_date)
        if end_date:
            stmt = stmt.where(TextNoteModel.timestamp <= end_date)

        result = await db.execute(stmt)
        return result.scalar()

    async def get_recent_by_user(
        self, db: AsyncSession, user_id: UUID, limit: int = 10
    ) -> List[TextNoteModel]:
        """Get the most recent text notes for a user."""
        stmt = (
            select(TextNoteModel)
            .where(TextNoteModel.user_id == str(user_id))
            .order_by(TextNoteModel.timestamp.desc())
            .limit(limit)
        )

        result = await db.execute(stmt)
        notes = result.scalars().all()

        # Convert names back to lists
        for note in notes:
            if note.names:
                try:
                    note.names = json.loads(note.names)
                except (json.JSONDecodeError, TypeError):
                    note.names = []

        return notes

    async def get_by_location(
        self,
        db: AsyncSession,
        user_id: UUID,
        longitude: Decimal,
        latitude: Decimal,
        radius_km: float = 1.0,
        skip: int = 0,
        limit: int = 100,
    ) -> List[TextNoteModel]:
        """Get text notes near specific coordinates using bounding box approximation."""
        # Simple bounding box approximation for performance
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        lat_delta = Decimal(radius_km / 111.0)
        lng_delta = Decimal(
            radius_km
            / (111.0 * abs(float(latitude.cos() if hasattr(latitude, "cos") else 1)))
        )

        stmt = (
            select(TextNoteModel)
            .where(
                and_(
                    TextNoteModel.user_id == str(user_id),
                    TextNoteModel.latitude.isnot(None),
                    TextNoteModel.longitude.isnot(None),
                    TextNoteModel.latitude.between(
                        latitude - lat_delta, latitude + lat_delta
                    ),
                    TextNoteModel.longitude.between(
                        longitude - lng_delta, longitude + lng_delta
                    ),
                )
            )
            .order_by(TextNoteModel.timestamp.desc())
            .offset(skip)
            .limit(limit)
        )

        result = await db.execute(stmt)
        notes = result.scalars().all()

        # Convert names back to lists
        for note in notes:
            if note.names:
                try:
                    note.names = json.loads(note.names)
                except (json.JSONDecodeError, TypeError):
                    note.names = []

        return notes

    async def search_by_address(
        self,
        db: AsyncSession,
        user_id: UUID,
        address_query: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[TextNoteModel]:
        """Search text notes by address using text search."""
        stmt = (
            select(TextNoteModel)
            .where(
                and_(
                    TextNoteModel.user_id == str(user_id),
                    TextNoteModel.address.isnot(None),
                    TextNoteModel.address.like(f"%{address_query}%"),
                )
            )
            .order_by(TextNoteModel.timestamp.desc())
            .offset(skip)
            .limit(limit)
        )

        result = await db.execute(stmt)
        notes = result.scalars().all()

        # Convert names back to lists
        for note in notes:
            if note.names:
                try:
                    note.names = json.loads(note.names)
                except (json.JSONDecodeError, TypeError):
                    note.names = []

        return notes

    async def get_with_location(
        self, db: AsyncSession, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[TextNoteModel]:
        """Get text notes that have location data."""
        stmt = (
            select(TextNoteModel)
            .where(
                and_(
                    TextNoteModel.user_id == str(user_id),
                    TextNoteModel.latitude.isnot(None),
                    TextNoteModel.longitude.isnot(None),
                )
            )
            .order_by(TextNoteModel.timestamp.desc())
            .offset(skip)
            .limit(limit)
        )

        result = await db.execute(stmt)
        notes = result.scalars().all()

        # Convert names back to lists
        for note in notes:
            if note.names:
                try:
                    note.names = json.loads(note.names)
                except (json.JSONDecodeError, TypeError):
                    note.names = []

        return notes


# Create a singleton instance
test_text_note_repository = TextNoteModelRepository()
