"""
Repository for media files with CRUD operations and specialized queries.
"""
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import MediaFile as DBMediaFile
from app.models.schemas import MediaFileCreate, FileType
from app.repositories.base import BaseRepository


class MediaFileRepository(BaseRepository[DBMediaFile, MediaFileCreate, None]):
    """
    Repository for media file operations with location-based filtering,
    name/tag searches, file type filtering, and proper user isolation.
    """
    
    def __init__(self):
        super().__init__(DBMediaFile)
    
    async def create(
        self, 
        db: AsyncSession, 
        *, 
        obj_in: MediaFileCreate, 
        user_id: UUID,
        file_path: str,
        file_size: Optional[int] = None
    ) -> DBMediaFile:
        """
        Create a new media file record.
        
        Args:
            db: Database session
            obj_in: Media file creation data
            user_id: ID of the user creating the file
            file_path: Path where the file is stored
            file_size: Size of the file in bytes
            
        Returns:
            Created media file record
        """
        db_obj = DBMediaFile(
            user_id=user_id,
            file_type=obj_in.file_type,
            file_path=file_path,
            original_filename=obj_in.original_filename,
            file_size=file_size,
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
    
    async def get(self, db: AsyncSession, id: UUID) -> Optional[DBMediaFile]:
        """
        Get a media file by ID.
        
        Args:
            db: Database session
            id: Media file ID
            
        Returns:
            Media file if found, None otherwise
        """
        stmt = select(DBMediaFile).where(DBMediaFile.id == id)
        result = await db.execute(stmt)
        return result.scalars().first()
    
    async def get_by_user(
        self, 
        db: AsyncSession, 
        user_id: UUID, 
        file_id: UUID
    ) -> Optional[DBMediaFile]:
        """
        Get a media file by ID for a specific user.
        
        Args:
            db: Database session
            user_id: User ID
            file_id: Media file ID
            
        Returns:
            Media file if found and belongs to user, None otherwise
        """
        stmt = select(DBMediaFile).where(
            and_(
                DBMediaFile.id == file_id,
                DBMediaFile.user_id == user_id
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
    ) -> List[DBMediaFile]:
        """
        Get multiple media files with optional filtering.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters to apply
            
        Returns:
            List of media files
        """
        stmt = select(DBMediaFile)
        
        # Apply filters if provided
        if filters:
            stmt = self._apply_filters(stmt, filters)
        
        stmt = stmt.offset(skip).limit(limit).order_by(DBMediaFile.timestamp.desc())
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_by_user_with_date_range(
        self,
        db: AsyncSession,
        user_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        file_type: Optional[FileType] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBMediaFile]:
        """
        Get media files for a user within a date range and optional file type filter.
        
        Args:
            db: Database session
            user_id: User ID
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            file_type: Optional file type filter
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of media files within the date range
        """
        stmt = select(DBMediaFile).where(DBMediaFile.user_id == user_id)
        
        # Apply date range filters
        if start_date:
            stmt = stmt.where(DBMediaFile.timestamp >= start_date)
        if end_date:
            stmt = stmt.where(DBMediaFile.timestamp <= end_date)
        
        # Apply file type filter
        if file_type:
            stmt = stmt.where(DBMediaFile.file_type == file_type.value)
        
        stmt = stmt.order_by(DBMediaFile.timestamp.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_by_file_type(
        self,
        db: AsyncSession,
        user_id: UUID,
        file_type: FileType,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBMediaFile]:
        """
        Get media files by file type for a specific user.
        
        Args:
            db: Database session
            user_id: User ID
            file_type: File type to filter by
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of media files of the specified type
        """
        stmt = select(DBMediaFile).where(
            and_(
                DBMediaFile.user_id == user_id,
                DBMediaFile.file_type == file_type.value
            )
        ).order_by(DBMediaFile.timestamp.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_by_location(
        self,
        db: AsyncSession,
        user_id: UUID,
        longitude: Decimal,
        latitude: Decimal,
        radius_km: float = 1.0,
        file_type: Optional[FileType] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBMediaFile]:
        """
        Get media files near specific coordinates using bounding box approximation.
        
        Args:
            db: Database session
            user_id: User ID
            longitude: Target longitude
            latitude: Target latitude
            radius_km: Search radius in kilometers
            file_type: Optional file type filter
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of media files within the specified radius
        """
        # Simple bounding box approximation for performance
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        lat_delta = Decimal(radius_km / 111.0)
        lng_delta = Decimal(radius_km / (111.0 * abs(float(latitude.cos() if hasattr(latitude, 'cos') else 1))))
        
        stmt = select(DBMediaFile).where(
            and_(
                DBMediaFile.user_id == user_id,
                DBMediaFile.latitude.isnot(None),
                DBMediaFile.longitude.isnot(None),
                DBMediaFile.latitude.between(latitude - lat_delta, latitude + lat_delta),
                DBMediaFile.longitude.between(longitude - lng_delta, longitude + lng_delta)
            )
        )
        
        # Apply file type filter if provided
        if file_type:
            stmt = stmt.where(DBMediaFile.file_type == file_type.value)
        
        stmt = stmt.order_by(DBMediaFile.timestamp.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def search_by_names(
        self,
        db: AsyncSession,
        user_id: UUID,
        search_terms: List[str],
        file_type: Optional[FileType] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBMediaFile]:
        """
        Search media files by names/tags using array operations.
        
        Args:
            db: Database session
            user_id: User ID
            search_terms: List of terms to search for in names array
            file_type: Optional file type filter
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of media files matching the search terms
        """
        stmt = select(DBMediaFile).where(DBMediaFile.user_id == user_id)
        
        # Use PostgreSQL array operations for case-insensitive search
        if search_terms:
            # Convert search terms to lowercase for case-insensitive search
            lower_terms = [term.lower() for term in search_terms]
            
            # Create conditions for each search term
            conditions = []
            for term in lower_terms:
                conditions.append(
                    func.lower(func.array_to_string(DBMediaFile.names, ' ')).contains(term)
                )
            
            if conditions:
                stmt = stmt.where(or_(*conditions))
        
        # Apply file type filter if provided
        if file_type:
            stmt = stmt.where(DBMediaFile.file_type == file_type.value)
        
        stmt = stmt.order_by(DBMediaFile.timestamp.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def search_by_address(
        self,
        db: AsyncSession,
        user_id: UUID,
        address_query: str,
        file_type: Optional[FileType] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBMediaFile]:
        """
        Search media files by address using text search.
        
        Args:
            db: Database session
            user_id: User ID
            address_query: Address search query
            file_type: Optional file type filter
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of media files matching the address query
        """
        stmt = select(DBMediaFile).where(
            and_(
                DBMediaFile.user_id == user_id,
                DBMediaFile.address.isnot(None),
                DBMediaFile.address.ilike(f"%{address_query}%")
            )
        )
        
        # Apply file type filter if provided
        if file_type:
            stmt = stmt.where(DBMediaFile.file_type == file_type.value)
        
        stmt = stmt.order_by(DBMediaFile.timestamp.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def search_by_filename(
        self,
        db: AsyncSession,
        user_id: UUID,
        filename_query: str,
        file_type: Optional[FileType] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBMediaFile]:
        """
        Search media files by original filename.
        
        Args:
            db: Database session
            user_id: User ID
            filename_query: Filename search query
            file_type: Optional file type filter
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of media files matching the filename query
        """
        stmt = select(DBMediaFile).where(
            and_(
                DBMediaFile.user_id == user_id,
                DBMediaFile.original_filename.isnot(None),
                DBMediaFile.original_filename.ilike(f"%{filename_query}%")
            )
        )
        
        # Apply file type filter if provided
        if file_type:
            stmt = stmt.where(DBMediaFile.file_type == file_type.value)
        
        stmt = stmt.order_by(DBMediaFile.timestamp.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_with_location(
        self,
        db: AsyncSession,
        user_id: UUID,
        file_type: Optional[FileType] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBMediaFile]:
        """
        Get media files that have location data.
        
        Args:
            db: Database session
            user_id: User ID
            file_type: Optional file type filter
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of media files with location data
        """
        stmt = select(DBMediaFile).where(
            and_(
                DBMediaFile.user_id == user_id,
                DBMediaFile.latitude.isnot(None),
                DBMediaFile.longitude.isnot(None)
            )
        )
        
        # Apply file type filter if provided
        if file_type:
            stmt = stmt.where(DBMediaFile.file_type == file_type.value)
        
        stmt = stmt.order_by(DBMediaFile.timestamp.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_large_files(
        self,
        db: AsyncSession,
        user_id: UUID,
        min_size_bytes: int,
        file_type: Optional[FileType] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBMediaFile]:
        """
        Get media files larger than a specified size.
        
        Args:
            db: Database session
            user_id: User ID
            min_size_bytes: Minimum file size in bytes
            file_type: Optional file type filter
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of media files larger than the specified size
        """
        stmt = select(DBMediaFile).where(
            and_(
                DBMediaFile.user_id == user_id,
                DBMediaFile.file_size.isnot(None),
                DBMediaFile.file_size >= min_size_bytes
            )
        )
        
        # Apply file type filter if provided
        if file_type:
            stmt = stmt.where(DBMediaFile.file_type == file_type.value)
        
        stmt = stmt.order_by(DBMediaFile.file_size.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def get_total_storage_used(
        self,
        db: AsyncSession,
        user_id: UUID,
        file_type: Optional[FileType] = None
    ) -> int:
        """
        Get total storage used by a user's media files.
        
        Args:
            db: Database session
            user_id: User ID
            file_type: Optional file type filter
            
        Returns:
            Total storage used in bytes
        """
        stmt = select(func.coalesce(func.sum(DBMediaFile.file_size), 0)).where(
            and_(
                DBMediaFile.user_id == user_id,
                DBMediaFile.file_size.isnot(None)
            )
        )
        
        # Apply file type filter if provided
        if file_type:
            stmt = stmt.where(DBMediaFile.file_type == file_type.value)
        
        result = await db.execute(stmt)
        return result.scalar() or 0
    
    async def update(
        self, 
        db: AsyncSession, 
        *, 
        db_obj: DBMediaFile, 
        obj_in: None
    ) -> DBMediaFile:
        """
        Update a media file. Media files are generally immutable after creation.
        
        Args:
            db: Database session
            db_obj: Existing media file object
            obj_in: Update data (not used for media files)
            
        Returns:
            Media file (unchanged)
        """
        # Media files are generally immutable after creation
        # This method is included for interface compliance
        return db_obj
    
    async def delete(self, db: AsyncSession, *, id: UUID) -> Optional[DBMediaFile]:
        """
        Delete a media file by ID.
        
        Args:
            db: Database session
            id: Media file ID
            
        Returns:
            Deleted media file if found, None otherwise
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
        file_id: UUID
    ) -> Optional[DBMediaFile]:
        """
        Delete a media file by ID for a specific user.
        
        Args:
            db: Database session
            user_id: User ID
            file_id: Media file ID
            
        Returns:
            Deleted media file if found and belongs to user, None otherwise
        """
        db_obj = await self.get_by_user(db, user_id, file_id)
        if db_obj:
            await db.delete(db_obj)
            await db.commit()
        return db_obj
    
    async def count(self, db: AsyncSession, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count media files with optional filtering.
        
        Args:
            db: Database session
            filters: Optional filters to apply
            
        Returns:
            Count of matching media files
        """
        stmt = select(func.count(DBMediaFile.id))
        
        if filters:
            stmt = self._apply_filters(stmt, filters)
        
        result = await db.execute(stmt)
        return result.scalar()
    
    async def count_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        file_type: Optional[FileType] = None
    ) -> int:
        """
        Count media files for a user within a date range and optional file type filter.
        
        Args:
            db: Database session
            user_id: User ID
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            file_type: Optional file type filter
            
        Returns:
            Count of matching media files
        """
        stmt = select(func.count(DBMediaFile.id)).where(DBMediaFile.user_id == user_id)
        
        if start_date:
            stmt = stmt.where(DBMediaFile.timestamp >= start_date)
        if end_date:
            stmt = stmt.where(DBMediaFile.timestamp <= end_date)
        if file_type:
            stmt = stmt.where(DBMediaFile.file_type == file_type.value)
        
        result = await db.execute(stmt)
        return result.scalar()
    
    async def get_recent_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
        file_type: Optional[FileType] = None,
        limit: int = 10
    ) -> List[DBMediaFile]:
        """
        Get the most recent media files for a user.
        
        Args:
            db: Database session
            user_id: User ID
            file_type: Optional file type filter
            limit: Maximum number of records to return
            
        Returns:
            List of recent media files
        """
        stmt = select(DBMediaFile).where(DBMediaFile.user_id == user_id)
        
        # Apply file type filter if provided
        if file_type:
            stmt = stmt.where(DBMediaFile.file_type == file_type.value)
        
        stmt = stmt.order_by(DBMediaFile.timestamp.desc()).limit(limit)
        
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
            stmt = stmt.where(DBMediaFile.user_id == filters["user_id"])
        
        if "file_type" in filters:
            stmt = stmt.where(DBMediaFile.file_type == filters["file_type"])
        
        if "start_date" in filters:
            stmt = stmt.where(DBMediaFile.timestamp >= filters["start_date"])
        
        if "end_date" in filters:
            stmt = stmt.where(DBMediaFile.timestamp <= filters["end_date"])
        
        if "has_location" in filters:
            if filters["has_location"]:
                stmt = stmt.where(
                    and_(
                        DBMediaFile.latitude.isnot(None),
                        DBMediaFile.longitude.isnot(None)
                    )
                )
            else:
                stmt = stmt.where(
                    or_(
                        DBMediaFile.latitude.is_(None),
                        DBMediaFile.longitude.is_(None)
                    )
                )
        
        if "address_contains" in filters:
            stmt = stmt.where(
                and_(
                    DBMediaFile.address.isnot(None),
                    DBMediaFile.address.ilike(f"%{filters['address_contains']}%")
                )
            )
        
        if "filename_contains" in filters:
            stmt = stmt.where(
                and_(
                    DBMediaFile.original_filename.isnot(None),
                    DBMediaFile.original_filename.ilike(f"%{filters['filename_contains']}%")
                )
            )
        
        if "names_contain" in filters:
            # Search for any of the provided terms in the names array
            search_terms = filters["names_contain"]
            if isinstance(search_terms, str):
                search_terms = [search_terms]
            
            conditions = []
            for term in search_terms:
                conditions.append(
                    func.lower(func.array_to_string(DBMediaFile.names, ' ')).contains(term.lower())
                )
            
            if conditions:
                stmt = stmt.where(or_(*conditions))
        
        if "min_file_size" in filters:
            stmt = stmt.where(
                and_(
                    DBMediaFile.file_size.isnot(None),
                    DBMediaFile.file_size >= filters["min_file_size"]
                )
            )
        
        if "max_file_size" in filters:
            stmt = stmt.where(
                and_(
                    DBMediaFile.file_size.isnot(None),
                    DBMediaFile.file_size <= filters["max_file_size"]
                )
            )
        
        return stmt


# Create a singleton instance
media_file_repository = MediaFileRepository()