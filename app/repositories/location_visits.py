"""
Repository for location visits with CRUD operations and specialized queries.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.database import LocationVisit as DBLocationVisit
from app.models.schemas import LocationVisitCreate, LocationVisitUpdate
from app.repositories.base import BaseRepository


class LocationVisitRepository(
    BaseRepository[DBLocationVisit, LocationVisitCreate, LocationVisitUpdate]
):
    """
    Repository for location visit operations with array field support,
    date range filtering, and coordinate-based queries.
    """

    def __init__(self):
        super().__init__(DBLocationVisit)

    async def create(
        self, db: AsyncSession, *, obj_in: LocationVisitCreate, user_id: UUID
    ) -> DBLocationVisit:
        """
        Create a new location visit.

        Args:
            db: Database session
            obj_in: Location visit creation data
            user_id: ID of the user creating the visit

        Returns:
            Created location visit
        """
        db_obj = DBLocationVisit(
            user_id=user_id,
            longitude=obj_in.longitude,
            latitude=obj_in.latitude,
            address=obj_in.address,
            names=obj_in.names,
            visit_time=obj_in.visit_time,
            duration=obj_in.duration,
            description=obj_in.description,
        )

        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)

        return db_obj

    async def get(self, db: AsyncSession, id: UUID) -> Optional[DBLocationVisit]:
        """
        Get a location visit by ID.

        Args:
            db: Database session
            id: Location visit ID

        Returns:
            Location visit if found, None otherwise
        """
        stmt = select(DBLocationVisit).where(DBLocationVisit.id == id)
        result = await db.execute(stmt)
        return result.scalars().first()

    async def get_by_user(
        self, db: AsyncSession, user_id: UUID, visit_id: UUID
    ) -> Optional[DBLocationVisit]:
        """
        Get a location visit by ID for a specific user.

        Args:
            db: Database session
            user_id: User ID
            visit_id: Location visit ID

        Returns:
            Location visit if found and belongs to user, None otherwise
        """
        stmt = select(DBLocationVisit).where(
            and_(DBLocationVisit.id == visit_id, DBLocationVisit.user_id == user_id)
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
    ) -> List[DBLocationVisit]:
        """
        Get multiple location visits with optional filtering.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters to apply

        Returns:
            List of location visits
        """
        stmt = select(DBLocationVisit)

        # Apply filters if provided
        if filters:
            stmt = self._apply_filters(stmt, filters)

        stmt = (
            stmt.offset(skip).limit(limit).order_by(DBLocationVisit.visit_time.desc())
        )

        result = await db.execute(stmt)
        return result.scalars().all()

    async def get_by_user_with_date_range(
        self,
        db: AsyncSession,
        user_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[DBLocationVisit]:
        """
        Get location visits for a user within a date range.

        Args:
            db: Database session
            user_id: User ID
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of location visits within the date range
        """
        stmt = select(DBLocationVisit).where(DBLocationVisit.user_id == user_id)

        # Apply date range filters
        if start_date:
            stmt = stmt.where(DBLocationVisit.visit_time >= start_date)
        if end_date:
            stmt = stmt.where(DBLocationVisit.visit_time <= end_date)

        stmt = (
            stmt.order_by(DBLocationVisit.visit_time.desc()).offset(skip).limit(limit)
        )

        result = await db.execute(stmt)
        return result.scalars().all()

    async def get_by_coordinates(
        self,
        db: AsyncSession,
        user_id: UUID,
        longitude: Decimal,
        latitude: Decimal,
        radius_km: float = 1.0,
        skip: int = 0,
        limit: int = 100,
    ) -> List[DBLocationVisit]:
        """
        Get location visits near specific coordinates using Haversine formula approximation.

        Args:
            db: Database session
            user_id: User ID
            longitude: Target longitude
            latitude: Target latitude
            radius_km: Search radius in kilometers
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of location visits within the specified radius
        """
        # Simple bounding box approximation for performance
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        lat_delta = Decimal(radius_km / 111.0)
        lng_delta = Decimal(
            radius_km
            / (111.0 * abs(float(latitude.cos() if hasattr(latitude, "cos") else 1)))
        )

        stmt = (
            select(DBLocationVisit)
            .where(
                and_(
                    DBLocationVisit.user_id == user_id,
                    DBLocationVisit.latitude.between(
                        latitude - lat_delta, latitude + lat_delta
                    ),
                    DBLocationVisit.longitude.between(
                        longitude - lng_delta, longitude + lng_delta
                    ),
                )
            )
            .order_by(DBLocationVisit.visit_time.desc())
            .offset(skip)
            .limit(limit)
        )

        result = await db.execute(stmt)
        return result.scalars().all()

    async def search_by_names(
        self,
        db: AsyncSession,
        user_id: UUID,
        search_terms: List[str],
        skip: int = 0,
        limit: int = 100,
    ) -> List[DBLocationVisit]:
        """
        Search location visits by names/tags using array operations.

        Args:
            db: Database session
            user_id: User ID
            search_terms: List of terms to search for in names array
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of location visits matching the search terms
        """
        stmt = select(DBLocationVisit).where(DBLocationVisit.user_id == user_id)

        # Use PostgreSQL array overlap operator to find matches
        if search_terms:
            # Convert search terms to lowercase for case-insensitive search
            lower_terms = [term.lower() for term in search_terms]

            # Use array overlap operator (&&) with case-insensitive comparison
            stmt = stmt.where(
                func.lower(func.array_to_string(DBLocationVisit.names, " ")).contains(
                    func.any_(lower_terms)
                )
            )

        stmt = (
            stmt.order_by(DBLocationVisit.visit_time.desc()).offset(skip).limit(limit)
        )

        result = await db.execute(stmt)
        return result.scalars().all()

    async def search_by_address(
        self,
        db: AsyncSession,
        user_id: UUID,
        address_query: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[DBLocationVisit]:
        """
        Search location visits by address using text search.

        Args:
            db: Database session
            user_id: User ID
            address_query: Address search query
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of location visits matching the address query
        """
        stmt = (
            select(DBLocationVisit)
            .where(
                and_(
                    DBLocationVisit.user_id == user_id,
                    DBLocationVisit.address.ilike(f"%{address_query}%"),
                )
            )
            .order_by(DBLocationVisit.visit_time.desc())
            .offset(skip)
            .limit(limit)
        )

        result = await db.execute(stmt)
        return result.scalars().all()

    async def update(
        self, db: AsyncSession, *, db_obj: DBLocationVisit, obj_in: LocationVisitUpdate
    ) -> DBLocationVisit:
        """
        Update a location visit with support for descriptions and names.

        Args:
            db: Database session
            db_obj: Existing location visit object
            obj_in: Update data

        Returns:
            Updated location visit
        """
        update_data = obj_in.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            setattr(db_obj, field, value)

        await db.commit()
        await db.refresh(db_obj)

        return db_obj

    async def update_by_id(
        self,
        db: AsyncSession,
        user_id: UUID,
        visit_id: UUID,
        obj_in: LocationVisitUpdate,
    ) -> Optional[DBLocationVisit]:
        """
        Update a location visit by ID for a specific user.

        Args:
            db: Database session
            user_id: User ID
            visit_id: Location visit ID
            obj_in: Update data

        Returns:
            Updated location visit if found and belongs to user, None otherwise
        """
        # First get the existing record
        db_obj = await self.get_by_user(db, user_id, visit_id)
        if not db_obj:
            return None

        return await self.update(db, db_obj=db_obj, obj_in=obj_in)

    async def delete(self, db: AsyncSession, *, id: UUID) -> Optional[DBLocationVisit]:
        """
        Delete a location visit by ID.

        Args:
            db: Database session
            id: Location visit ID

        Returns:
            Deleted location visit if found, None otherwise
        """
        db_obj = await self.get(db, id)
        if db_obj:
            await db.delete(db_obj)
            await db.commit()
        return db_obj

    async def delete_by_user(
        self, db: AsyncSession, user_id: UUID, visit_id: UUID
    ) -> Optional[DBLocationVisit]:
        """
        Delete a location visit by ID for a specific user.

        Args:
            db: Database session
            user_id: User ID
            visit_id: Location visit ID

        Returns:
            Deleted location visit if found and belongs to user, None otherwise
        """
        db_obj = await self.get_by_user(db, user_id, visit_id)
        if db_obj:
            await db.delete(db_obj)
            await db.commit()
        return db_obj

    async def count(
        self, db: AsyncSession, filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count location visits with optional filtering.

        Args:
            db: Database session
            filters: Optional filters to apply

        Returns:
            Count of matching location visits
        """
        stmt = select(func.count(DBLocationVisit.id))

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
    ) -> int:
        """
        Count location visits for a user within a date range.

        Args:
            db: Database session
            user_id: User ID
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            Count of matching location visits
        """
        stmt = select(func.count(DBLocationVisit.id)).where(
            DBLocationVisit.user_id == user_id
        )

        if start_date:
            stmt = stmt.where(DBLocationVisit.visit_time >= start_date)
        if end_date:
            stmt = stmt.where(DBLocationVisit.visit_time <= end_date)

        result = await db.execute(stmt)
        return result.scalar()

    async def get_recent_by_user(
        self, db: AsyncSession, user_id: UUID, limit: int = 10
    ) -> List[DBLocationVisit]:
        """
        Get the most recent location visits for a user.

        Args:
            db: Database session
            user_id: User ID
            limit: Maximum number of records to return

        Returns:
            List of recent location visits
        """
        stmt = (
            select(DBLocationVisit)
            .where(DBLocationVisit.user_id == user_id)
            .order_by(DBLocationVisit.visit_time.desc())
            .limit(limit)
        )

        result = await db.execute(stmt)
        return result.scalars().all()

    async def get_visits_with_descriptions(
        self, db: AsyncSession, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[DBLocationVisit]:
        """
        Get location visits that have descriptions.

        Args:
            db: Database session
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of location visits with descriptions
        """
        stmt = (
            select(DBLocationVisit)
            .where(
                and_(
                    DBLocationVisit.user_id == user_id,
                    DBLocationVisit.description.isnot(None),
                    DBLocationVisit.description != "",
                )
            )
            .order_by(DBLocationVisit.visit_time.desc())
            .offset(skip)
            .limit(limit)
        )

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
            stmt = stmt.where(DBLocationVisit.user_id == filters["user_id"])

        if "start_date" in filters:
            stmt = stmt.where(DBLocationVisit.visit_time >= filters["start_date"])

        if "end_date" in filters:
            stmt = stmt.where(DBLocationVisit.visit_time <= filters["end_date"])

        if "has_description" in filters:
            if filters["has_description"]:
                stmt = stmt.where(
                    and_(
                        DBLocationVisit.description.isnot(None),
                        DBLocationVisit.description != "",
                    )
                )
            else:
                stmt = stmt.where(
                    or_(
                        DBLocationVisit.description.is_(None),
                        DBLocationVisit.description == "",
                    )
                )

        if "min_duration" in filters:
            stmt = stmt.where(DBLocationVisit.duration >= filters["min_duration"])

        if "max_duration" in filters:
            stmt = stmt.where(DBLocationVisit.duration <= filters["max_duration"])

        if "address_contains" in filters:
            stmt = stmt.where(
                DBLocationVisit.address.ilike(f"%{filters['address_contains']}%")
            )

        if "names_contain" in filters:
            # Search for any of the provided terms in the names array
            search_terms = filters["names_contain"]
            if isinstance(search_terms, str):
                search_terms = [search_terms]

            conditions = []
            for term in search_terms:
                conditions.append(
                    func.lower(
                        func.array_to_string(DBLocationVisit.names, " ")
                    ).contains(term.lower())
                )

            if conditions:
                stmt = stmt.where(or_(*conditions))

        return stmt


# Create a singleton instance
location_visit_repository = LocationVisitRepository()
