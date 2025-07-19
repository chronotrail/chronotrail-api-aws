"""
SQLite-compatible repository for testing location visits.
"""
import json
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import LocationVisitModel, UserModel
from app.models.schemas import LocationVisitCreate, LocationVisitUpdate


class LocationVisitModelRepository:
    """
    Test repository for location visit operations with SQLite compatibility.
    """
    
    def __init__(self):
        self.model = LocationVisitModel
    
    async def create(
        self, 
        db: AsyncSession, 
        *, 
        obj_in: LocationVisitCreate, 
        user_id: UUID
    ) -> LocationVisitModel:
        """Create a new location visit."""
        # Convert names list to JSON string for SQLite
        names_json = json.dumps(obj_in.names) if obj_in.names else None
        
        # Use no_autoflush to prevent premature flushing
        with db.no_autoflush:
            db_obj = LocationVisitModel(
                user_id=str(user_id),
                longitude=obj_in.longitude,
                latitude=obj_in.latitude,
                address=obj_in.address,
                names=names_json,
                visit_time=obj_in.visit_time,
                duration=obj_in.duration,
                description=obj_in.description
            )
            
            db.add(db_obj)
            await db.commit()
            await db.refresh(db_obj)
            
            # Convert back to list for response
            if db_obj.names:
                try:
                    db_obj.names = json.loads(db_obj.names)
                except (json.JSONDecodeError, TypeError):
                    db_obj.names = []
        
        return db_obj
    
    async def get(self, db: AsyncSession, id: UUID) -> Optional[LocationVisitModel]:
        """Get a location visit by ID."""
        with db.no_autoflush:
            stmt = select(LocationVisitModel).where(LocationVisitModel.id == str(id))
            result = await db.execute(stmt)
            visit = result.scalars().first()
            
            if visit and visit.names:
                try:
                    visit.names = json.loads(visit.names)
                except (json.JSONDecodeError, TypeError):
                    visit.names = []
            
            return visit
    
    async def get_by_user(
        self, 
        db: AsyncSession, 
        user_id: UUID, 
        visit_id: UUID
    ) -> Optional[LocationVisitModel]:
        """Get a location visit by ID for a specific user."""
        with db.no_autoflush:
            stmt = select(LocationVisitModel).where(
                and_(
                    LocationVisitModel.id == str(visit_id),
                    LocationVisitModel.user_id == str(user_id)
                )
            )
            result = await db.execute(stmt)
            visit = result.scalars().first()
            
            if visit and visit.names:
                try:
                    visit.names = json.loads(visit.names)
                except (json.JSONDecodeError, TypeError):
                    visit.names = []
            
            return visit
    
    async def get_by_user_with_date_range(
        self,
        db: AsyncSession,
        user_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[LocationVisitModel]:
        """Get location visits for a user within a date range."""
        stmt = select(LocationVisitModel).where(LocationVisitModel.user_id == str(user_id))
        
        if start_date:
            stmt = stmt.where(LocationVisitModel.visit_time >= start_date)
        if end_date:
            stmt = stmt.where(LocationVisitModel.visit_time <= end_date)
        
        stmt = stmt.order_by(LocationVisitModel.visit_time.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        visits = result.scalars().all()
        
        # Convert names back to lists
        for visit in visits:
            if visit.names:
                visit.names = json.loads(visit.names)
        
        return visits
    
    async def search_by_names(
        self,
        db: AsyncSession,
        user_id: UUID,
        search_terms: List[str],
        skip: int = 0,
        limit: int = 100
    ) -> List[LocationVisitModel]:
        """Search location visits by names/tags."""
        stmt = select(LocationVisitModel).where(LocationVisitModel.user_id == str(user_id))
        
        if search_terms:
            # For SQLite, search within the JSON string
            conditions = []
            for term in search_terms:
                conditions.append(LocationVisitModel.names.like(f'%"{term}"%'))
            
            if conditions:
                stmt = stmt.where(or_(*conditions))
        
        stmt = stmt.order_by(LocationVisitModel.visit_time.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        visits = result.scalars().all()
        
        # Convert names back to lists
        for visit in visits:
            if visit.names:
                visit.names = json.loads(visit.names)
        
        return visits
    
    async def search_by_address(
        self,
        db: AsyncSession,
        user_id: UUID,
        address_query: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[LocationVisitModel]:
        """Search location visits by address."""
        stmt = select(LocationVisitModel).where(
            and_(
                LocationVisitModel.user_id == str(user_id),
                LocationVisitModel.address.like(f"%{address_query}%")
            )
        ).order_by(LocationVisitModel.visit_time.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        visits = result.scalars().all()
        
        # Convert names back to lists
        for visit in visits:
            if visit.names:
                visit.names = json.loads(visit.names)
        
        return visits
    
    async def update_by_id(
        self,
        db: AsyncSession,
        user_id: UUID,
        visit_id: UUID,
        obj_in: LocationVisitUpdate
    ) -> Optional[LocationVisitModel]:
        """Update a location visit by ID for a specific user."""
        # Use no_autoflush to prevent premature flushing of pending changes
        with db.no_autoflush:
            # First get the existing record
            db_obj = await self.get_by_user(db, user_id, visit_id)
            if not db_obj:
                return None
            
            update_data = obj_in.model_dump(exclude_unset=True)
            
            # Convert names list to JSON string if provided
            if 'names' in update_data and update_data['names'] is not None:
                update_data['names'] = json.dumps(update_data['names'])
            
            # Update the record
            stmt = update(LocationVisitModel).where(
                and_(
                    LocationVisitModel.id == str(visit_id),
                    LocationVisitModel.user_id == str(user_id)
                )
            ).values(**update_data)
            
            await db.execute(stmt)
            await db.commit()
            
            # Return the updated record
            return await self.get_by_user(db, user_id, visit_id)
    
    async def delete_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
        visit_id: UUID
    ) -> Optional[LocationVisitModel]:
        """Delete a location visit by ID for a specific user."""
        # Use no_autoflush to prevent premature flushing of pending changes
        with db.no_autoflush:
            db_obj = await self.get_by_user(db, user_id, visit_id)
            if db_obj:
                stmt = delete(LocationVisitModel).where(
                    and_(
                        LocationVisitModel.id == str(visit_id),
                        LocationVisitModel.user_id == str(user_id)
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
        end_date: Optional[datetime] = None
    ) -> int:
        """Count location visits for a user within a date range."""
        stmt = select(func.count(LocationVisitModel.id)).where(LocationVisitModel.user_id == str(user_id))
        
        if start_date:
            stmt = stmt.where(LocationVisitModel.visit_time >= start_date)
        if end_date:
            stmt = stmt.where(LocationVisitModel.visit_time <= end_date)
        
        result = await db.execute(stmt)
        return result.scalar()
    
    async def get_recent_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
        limit: int = 10
    ) -> List[LocationVisitModel]:
        """Get the most recent location visits for a user."""
        stmt = select(LocationVisitModel).where(
            LocationVisitModel.user_id == str(user_id)
        ).order_by(LocationVisitModel.visit_time.desc()).limit(limit)
        
        result = await db.execute(stmt)
        visits = result.scalars().all()
        
        # Convert names back to lists
        for visit in visits:
            if visit.names:
                visit.names = json.loads(visit.names)
        
        return visits
    
    async def get_visits_with_descriptions(
        self,
        db: AsyncSession,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[LocationVisitModel]:
        """Get location visits that have descriptions."""
        stmt = select(LocationVisitModel).where(
            and_(
                LocationVisitModel.user_id == str(user_id),
                LocationVisitModel.description.isnot(None),
                LocationVisitModel.description != ""
            )
        ).order_by(LocationVisitModel.visit_time.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        visits = result.scalars().all()
        
        # Convert names back to lists
        for visit in visits:
            if visit.names:
                visit.names = json.loads(visit.names)
        
        return visits
    
    async def get_by_coordinates(
        self,
        db: AsyncSession,
        user_id: UUID,
        longitude: Decimal,
        latitude: Decimal,
        radius_km: float = 1.0,
        skip: int = 0,
        limit: int = 100
    ) -> List[LocationVisitModel]:
        """Get location visits near specific coordinates."""
        # Simple bounding box approximation
        lat_delta = Decimal(radius_km / 111.0)
        lng_delta = Decimal(radius_km / 111.0)  # Simplified for testing
        
        stmt = select(LocationVisitModel).where(
            and_(
                LocationVisitModel.user_id == str(user_id),
                LocationVisitModel.latitude.between(latitude - lat_delta, latitude + lat_delta),
                LocationVisitModel.longitude.between(longitude - lng_delta, longitude + lng_delta)
            )
        ).order_by(LocationVisitModel.visit_time.desc()).offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        visits = result.scalars().all()
        
        # Convert names back to lists
        for visit in visits:
            if visit.names:
                visit.names = json.loads(visit.names)
        
        return visits


# Create a singleton instance
test_location_visit_repository = LocationVisitModelRepository()