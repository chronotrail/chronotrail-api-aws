"""
Repository for text notes with CRUD operations and specialized queries.
"""
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import TextNote as DBTextNote
from app.models.schemas import TextNoteCreate
from app.repositories.base import BaseRepository


class TextNoteRepository(BaseRepository[DBTextNote, TextNoteCreate, None]):
    """
    Repository for text note operations with location-based filtering,
    name/tag searches, and proper user isolation.
    """
    
    def __init__(self):
        super().__init__(DBTextNote)
    
    async def create(
        self, 
        db: AsyncSession, 
        *, 
        obj_in: TextNoteCreate, 
        user_id: UUID
    ) -> DBTextNote:
        """
        Create a new text note.
        
        Args:
            db: Database session
            obj_in: Text note creation data
            user_id: ID of the user creating the note
            
        Returns:
            Created text note
        """
        db_obj = DBTextNote(
            user_id=user_id,
            text_content=obj_in.text_content,
            timestamp=obj_in.timestamp,
            longitude=obj_in.longitude,
            latitude=obj_in.latitude,
            address=obj_in.address,
            names=obj_in.names
        )
        
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        
        return db_obj
    
    async def get(self, db: AsyncSession, id: UUID) -> Optional[DBTextNote]:
        """
        Get a text note by ID.
        
        Args:
            db: Database session
            id: Text note ID
            
        Returns:
            Text note if found, None otherwise
        """
        stmt = select(DBTextNote).where(DBTextNote.id == id)
        result = await db.execute(stmt)
        return result.scalars().first()
    
    async def get_by_user(
        self, 
        db: AsyncSession, 
        user_id: UUID, 
        note_id: UUID
    ) -> Optional[DBTextNote]:
        """
        Get a text note by ID for a specific user.
        
        Args:
            db: Database session
            user_id: User ID
            note_id: Text note ID
            
        Returns:
            Text note if found and belongs to user, None otherwise
        """
        stmt = select(DBTextNote).where(
            and_(
                DBTextNote.id == note_id,
                DBTextNote.user_id == user_id
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
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DBTextNote]:
        """
        Get multiple text notes with optional filtering.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters to apply
            
        Returns:
            List of text notes
        """
        stmt = select(DBTextNote)
        
        # Apply filters if provided
        if filters:
            stmt = self._apply_filters(stmt, filters)
        
        stmt = stmt.offset(skip).limit(limit).order_by(DBTextNote.timestamp.desc())
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_by_user_with_date_range(
        self,
        db: AsyncSession,
        user_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBTextNote]:
        """
        Get text notes for a user within a date range.
        
        Args:
            db: Database session
            user_id: User ID
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of text notes within the date range
        """
        stmt = select(DBTextNote).where(DBTextNote.user_id == user_id)
        
        # Apply date range filters
        if start_date:
            stmt = stmt.where(DBTextNote.timestamp >= start_date)
        if end_date:
            stmt = stmt.where(DBTextNote.timestamp <= end_date)
        
        stmt = stmt.order_by(DBTextNote.timestamp.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_by_location(
        self,
        db: AsyncSession,
        user_id: UUID,
        longitude: Decimal,
        latitude: Decimal,
        radius_km: float = 1.0,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBTextNote]:
        """
        Get text notes near specific coordinates using bounding box approximation.
        
        Args:
            db: Database session
            user_id: User ID
            longitude: Target longitude
            latitude: Target latitude
            radius_km: Search radius in kilometers
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of text notes within the specified radius
        """
        # Simple bounding box approximation for performance
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        lat_delta = Decimal(radius_km / 111.0)
        lng_delta = Decimal(radius_km / (111.0 * abs(float(latitude.cos() if hasattr(latitude, 'cos') else 1))))
        
        stmt = select(DBTextNote).where(
            and_(
                DBTextNote.user_id == user_id,
                DBTextNote.latitude.isnot(None),
                DBTextNote.longitude.isnot(None),
                DBTextNote.latitude.between(latitude - lat_delta, latitude + lat_delta),
                DBTextNote.longitude.between(longitude - lng_delta, longitude + lng_delta)
            )
        ).order_by(DBTextNote.timestamp.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def search_by_names(
        self,
        db: AsyncSession,
        user_id: UUID,
        search_terms: List[str],
        skip: int = 0,
        limit: int = 100
    ) -> List[DBTextNote]:
        """
        Search text notes by names/tags using array operations.
        
        Args:
            db: Database session
            user_id: User ID
            search_terms: List of terms to search for in names array
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of text notes matching the search terms
        """
        stmt = select(DBTextNote).where(DBTextNote.user_id == user_id)
        
        # Use PostgreSQL array operations for case-insensitive search
        if search_terms:
            # Convert search terms to lowercase for case-insensitive search
            lower_terms = [term.lower() for term in search_terms]
            
            # Create conditions for each search term
            conditions = []
            for term in lower_terms:
                conditions.append(
                    func.lower(func.array_to_string(DBTextNote.names, ' ')).contains(term)
                )
            
            if conditions:
                stmt = stmt.where(or_(*conditions))
        
        stmt = stmt.order_by(DBTextNote.timestamp.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def search_by_address(
        self,
        db: AsyncSession,
        user_id: UUID,
        address_query: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBTextNote]:
        """
        Search text notes by address using text search.
        
        Args:
            db: Database session
            user_id: User ID
            address_query: Address search query
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of text notes matching the address query
        """
        stmt = select(DBTextNote).where(
            and_(
                DBTextNote.user_id == user_id,
                DBTextNote.address.isnot(None),
                DBTextNote.address.ilike(f"%{address_query}%")
            )
        ).order_by(DBTextNote.timestamp.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def search_by_content(
        self,
        db: AsyncSession,
        user_id: UUID,
        content_query: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBTextNote]:
        """
        Search text notes by content using text search.
        
        Args:
            db: Database session
            user_id: User ID
            content_query: Content search query
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of text notes matching the content query
        """
        stmt = select(DBTextNote).where(
            and_(
                DBTextNote.user_id == user_id,
                DBTextNote.text_content.ilike(f"%{content_query}%")
            )
        ).order_by(DBTextNote.timestamp.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_with_location(
        self,
        db: AsyncSession,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBTextNote]:
        """
        Get text notes that have location data.
        
        Args:
            db: Database session
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of text notes with location data
        """
        stmt = select(DBTextNote).where(
            and_(
                DBTextNote.user_id == user_id,
                DBTextNote.latitude.isnot(None),
                DBTextNote.longitude.isnot(None)
            )
        ).order_by(DBTextNote.timestamp.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def update(
        self, 
        db: AsyncSession, 
        *, 
        db_obj: DBTextNote, 
        obj_in: None
    ) -> DBTextNote:
        """
        Update a text note. Text notes are generally immutable after creation.
        
        Args:
            db: Database session
            db_obj: Existing text note object
            obj_in: Update data (not used for text notes)
            
        Returns:
            Text note (unchanged)
        """
        # Text notes are generally immutable after creation
        # This method is included for interface compliance
        return db_obj
    
    async def delete(self, db: AsyncSession, *, id: UUID) -> Optional[DBTextNote]:
        """
        Delete a text note by ID.
        
        Args:
            db: Database session
            id: Text note ID
            
        Returns:
            Deleted text note if found, None otherwise
        """
        db_obj = await self.get(db, id)
        if db_obj:
            await db.delete(db_obj)
            await db.commit()
        return db_obj
    
    async def delete_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
        note_id: UUID
    ) -> Optional[DBTextNote]:
        """
        Delete a text note by ID for a specific user.
        
        Args:
            db: Database session
            user_id: User ID
            note_id: Text note ID
            
        Returns:
            Deleted text note if found and belongs to user, None otherwise
        """
        db_obj = await self.get_by_user(db, user_id, note_id)
        if db_obj:
            await db.delete(db_obj)
            await db.commit()
        return db_obj
    
    async def count(self, db: AsyncSession, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count text notes with optional filtering.
        
        Args:
            db: Database session
            filters: Optional filters to apply
            
        Returns:
            Count of matching text notes
        """
        stmt = select(func.count(DBTextNote.id))
        
        if filters:
            stmt = self._apply_filters(stmt, filters)
        
        result = await db.execute(stmt)
        return result.scalar()
    
    async def count_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """
        Count text notes for a user within a date range.
        
        Args:
            db: Database session
            user_id: User ID
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            
        Returns:
            Count of matching text notes
        """
        stmt = select(func.count(DBTextNote.id)).where(DBTextNote.user_id == user_id)
        
        if start_date:
            stmt = stmt.where(DBTextNote.timestamp >= start_date)
        if end_date:
            stmt = stmt.where(DBTextNote.timestamp <= end_date)
        
        result = await db.execute(stmt)
        return result.scalar()
    
    async def get_recent_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
        limit: int = 10
    ) -> List[DBTextNote]:
        """
        Get the most recent text notes for a user.
        
        Args:
            db: Database session
            user_id: User ID
            limit: Maximum number of records to return
            
        Returns:
            List of recent text notes
        """
        stmt = select(DBTextNote).where(
            DBTextNote.user_id == user_id
        ).order_by(DBTextNote.timestamp.desc()).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
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
            stmt = stmt.where(DBTextNote.user_id == filters["user_id"])
        
        if "start_date" in filters:
            stmt = stmt.where(DBTextNote.timestamp >= filters["start_date"])
        
        if "end_date" in filters:
            stmt = stmt.where(DBTextNote.timestamp <= filters["end_date"])
        
        if "has_location" in filters:
            if filters["has_location"]:
                stmt = stmt.where(
                    and_(
                        DBTextNote.latitude.isnot(None),
                        DBTextNote.longitude.isnot(None)
                    )
                )
            else:
                stmt = stmt.where(
                    or_(
                        DBTextNote.latitude.is_(None),
                        DBTextNote.longitude.is_(None)
                    )
                )
        
        if "address_contains" in filters:
            stmt = stmt.where(
                and_(
                    DBTextNote.address.isnot(None),
                    DBTextNote.address.ilike(f"%{filters['address_contains']}%")
                )
            )
        
        if "content_contains" in filters:
            stmt = stmt.where(DBTextNote.text_content.ilike(f"%{filters['content_contains']}%"))
        
        if "names_contain" in filters:
            # Search for any of the provided terms in the names array
            search_terms = filters["names_contain"]
            if isinstance(search_terms, str):
                search_terms = [search_terms]
            
            conditions = []
            for term in search_terms:
                conditions.append(
                    func.lower(func.array_to_string(DBTextNote.names, ' ')).contains(term.lower())
                )
            
            if conditions:
                stmt = stmt.where(or_(*conditions))
        
        return stmt


# Create a singleton instance
text_note_repository = TextNoteRepository()