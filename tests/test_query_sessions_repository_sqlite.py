"""
SQLite-compatible repository for testing query sessions.
"""
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import QuerySessionModel, UserModel
from app.models.schemas import QuerySessionCreate, QuerySessionUpdate


class QuerySessionModelRepository:
    """
    Test repository for query session operations with SQLite compatibility.
    """
    
    def __init__(self):
        self.model = QuerySessionModel
    
    async def create(
        self, 
        db: AsyncSession, 
        *, 
        obj_in: QuerySessionCreate, 
        **kwargs
    ) -> QuerySessionModel:
        """Create a new query session."""
        # Set default expiration to 1 hour from now
        expires_at = datetime.utcnow() + timedelta(hours=1)
        
        # Convert session_context dict to JSON string for SQLite
        context_json = json.dumps(obj_in.session_context) if obj_in.session_context else "{}"
        
        db_obj = QuerySessionModel(
            user_id=str(obj_in.user_id),
            session_context=context_json,
            expires_at=expires_at
        )
        
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        
        # Detach from session to prevent further tracking
        db.expunge(db_obj)
        
        # Convert session_context back to dict for response
        if db_obj.session_context:
            try:
                db_obj.session_context = json.loads(db_obj.session_context)
            except (json.JSONDecodeError, TypeError):
                db_obj.session_context = {}
        
        return db_obj
    
    async def get(self, db: AsyncSession, id: UUID) -> Optional[QuerySessionModel]:
        """Get a query session by ID."""
        stmt = select(QuerySessionModel).where(
            and_(
                QuerySessionModel.id == str(id),
                QuerySessionModel.expires_at > datetime.utcnow()
            )
        )
        result = await db.execute(stmt)
        session = result.scalars().first()
        
        if session and session.session_context:
            try:
                session.session_context = json.loads(session.session_context)
            except (json.JSONDecodeError, TypeError):
                session.session_context = {}
        
        return session
    
    async def get_by_user(
        self, 
        db: AsyncSession, 
        user_id: UUID, 
        session_id: UUID
    ) -> Optional[QuerySessionModel]:
        """Get a query session by ID for a specific user."""
        with db.no_autoflush:
            stmt = select(QuerySessionModel).where(
                and_(
                    QuerySessionModel.id == str(session_id),
                    QuerySessionModel.user_id == str(user_id),
                    QuerySessionModel.expires_at > datetime.utcnow()
                )
            )
            result = await db.execute(stmt)
            session = result.scalars().first()
            
            if session:
                # Refresh the object to get the latest data from the database
                await db.refresh(session)
                
                if session.session_context:
                    try:
                        session.session_context = json.loads(session.session_context)
                    except (json.JSONDecodeError, TypeError):
                        session.session_context = {}
                else:
                    session.session_context = {}
        
        return session
    
    async def get_multi(
        self, 
        db: AsyncSession, 
        *, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[QuerySessionModel]:
        """Get multiple query sessions with optional filtering."""
        with db.no_autoflush:
            stmt = select(QuerySessionModel).where(
                QuerySessionModel.expires_at > datetime.utcnow()
            )
            
            # Apply filters if provided
            if filters:
                stmt = self._apply_filters(stmt, filters)
            
            stmt = stmt.offset(skip).limit(limit).order_by(QuerySessionModel.updated_at.desc())
            
            result = await db.execute(stmt)
            sessions = result.scalars().all()
            
            # Convert session_context back to dicts
            for session in sessions:
                if session.session_context:
                    try:
                        session.session_context = json.loads(session.session_context)
                    except (json.JSONDecodeError, TypeError):
                        session.session_context = {}
        
        return sessions
    
    async def get_active_sessions_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
        skip: int = 0,
        limit: int = 10
    ) -> List[QuerySessionModel]:
        """Get active query sessions for a user."""
        with db.no_autoflush:
            stmt = select(QuerySessionModel).where(
                and_(
                    QuerySessionModel.user_id == str(user_id),
                    QuerySessionModel.expires_at > datetime.utcnow()
                )
            ).order_by(QuerySessionModel.updated_at.desc()).offset(skip).limit(limit)
            
            result = await db.execute(stmt)
            sessions = result.scalars().all()
            
            # Convert session_context back to dicts
            for session in sessions:
                if session.session_context:
                    try:
                        session.session_context = json.loads(session.session_context)
                    except (json.JSONDecodeError, TypeError):
                        session.session_context = {}
        
        return sessions
    
    async def update(
        self, 
        db: AsyncSession, 
        *, 
        db_obj: QuerySessionModel, 
        obj_in: QuerySessionUpdate
    ) -> QuerySessionModel:
        """Update a query session with new context or query information."""
        update_data = obj_in.model_dump(exclude_unset=True)
        
        # Always update the updated_at timestamp
        update_data["updated_at"] = datetime.utcnow()
        
        # Extend session expiration by 1 hour when updated
        update_data["expires_at"] = datetime.utcnow() + timedelta(hours=1)
        
        # Convert session_context to JSON string if provided
        if "session_context" in update_data and update_data["session_context"] is not None:
            update_data["session_context"] = json.dumps(update_data["session_context"])
        
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        
        # Convert session_context back to dict
        if db_obj.session_context:
            try:
                db_obj.session_context = json.loads(db_obj.session_context)
            except (json.JSONDecodeError, TypeError):
                db_obj.session_context = {}
        
        return db_obj
    
    async def update_context(
        self,
        db: AsyncSession,
        session_id: UUID,
        user_id: UUID,
        context_updates: Dict[str, Any],
        last_query: Optional[str] = None
    ) -> Optional[QuerySessionModel]:
        """Update session context with new conversation data."""
        db_obj = await self.get_by_user(db, user_id, session_id)
        if not db_obj:
            return None
        
        # Merge context updates with existing context
        current_context = db_obj.session_context or {}
        current_context.update(context_updates)
        
        update_data = QuerySessionUpdate(
            session_context=current_context,
            last_query=last_query
        )
        
        return await self.update(db, db_obj=db_obj, obj_in=update_data)
    
    async def add_media_reference(
        self,
        db: AsyncSession,
        session_id: UUID,
        user_id: UUID,
        media_id: UUID,
        media_type: str,
        description: str
    ) -> Optional[QuerySessionModel]:
        """Add a media reference to the session context."""
        # Get the session with fresh data to ensure we have the latest context
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
            "referenced_at": datetime.utcnow().isoformat()
        }
        
        media_references.append(media_ref)
        current_context["media_references"] = media_references
        
        update_data = QuerySessionUpdate(session_context=current_context)
        return await self.update(db, db_obj=db_obj, obj_in=update_data)
    
    async def get_media_references(
        self,
        db: AsyncSession,
        session_id: UUID,
        user_id: UUID
    ) -> List[Dict[str, Any]]:
        """Get all media references from a session."""
        db_obj = await self.get_by_user(db, user_id, session_id)
        if not db_obj or not db_obj.session_context:
            return []
        
        return db_obj.session_context.get("media_references", [])
    
    async def extend_expiration(
        self,
        db: AsyncSession,
        session_id: UUID,
        user_id: UUID,
        hours: int = 1
    ) -> Optional[QuerySessionModel]:
        """Extend the expiration time of a session."""
        db_obj = await self.get_by_user(db, user_id, session_id)
        if not db_obj:
            return None
        
        # Extend expiration
        new_expiration = datetime.utcnow() + timedelta(hours=hours)
        db_obj.expires_at = new_expiration
        db_obj.updated_at = datetime.utcnow()
        
        # Convert session_context to JSON string for SQLite before saving
        if isinstance(db_obj.session_context, dict):
            db_obj.session_context = json.dumps(db_obj.session_context)
        
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        
        # Convert session_context back to dict
        if db_obj.session_context:
            try:
                db_obj.session_context = json.loads(db_obj.session_context)
            except (json.JSONDecodeError, TypeError):
                db_obj.session_context = {}
        
        return db_obj
    
    async def delete(self, db: AsyncSession, *, id: UUID) -> Optional[QuerySessionModel]:
        """Delete a query session by ID."""
        db_obj = await self.get(db, id)
        if db_obj:
            stmt = delete(QuerySessionModel).where(QuerySessionModel.id == str(id))
            await db.execute(stmt)
            await db.commit()
        return db_obj
    
    async def delete_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
        session_id: UUID
    ) -> Optional[QuerySessionModel]:
        """Delete a query session by ID for a specific user."""
        with db.no_autoflush:
            db_obj = await self.get_by_user(db, user_id, session_id)
            if db_obj:
                stmt = delete(QuerySessionModel).where(
                    and_(
                        QuerySessionModel.id == str(session_id),
                        QuerySessionModel.user_id == str(user_id)
                    )
                )
                await db.execute(stmt)
                await db.commit()
        return db_obj
    
    async def cleanup_expired_sessions(self, db: AsyncSession) -> int:
        """Delete all expired query sessions."""
        # Get count of expired sessions first
        count_stmt = select(func.count(QuerySessionModel.id)).where(
            QuerySessionModel.expires_at <= datetime.utcnow()
        )
        result = await db.execute(count_stmt)
        expired_count = result.scalar()
        
        # Delete expired sessions
        delete_stmt = delete(QuerySessionModel).where(
            QuerySessionModel.expires_at <= datetime.utcnow()
        )
        await db.execute(delete_stmt)
        await db.commit()
        
        return expired_count
    
    async def cleanup_old_sessions_by_user(
        self,
        db: AsyncSession,
        user_id: UUID,
        keep_count: int = 10
    ) -> int:
        """Clean up old sessions for a user, keeping only the most recent ones."""
        with db.no_autoflush:
            # Get sessions to keep (most recent ones)
            keep_stmt = select(QuerySessionModel.id).where(
                QuerySessionModel.user_id == str(user_id)
            ).order_by(QuerySessionModel.updated_at.desc()).limit(keep_count)
            
            keep_result = await db.execute(keep_stmt)
            keep_ids = [row[0] for row in keep_result.fetchall()]
            
            if not keep_ids:
                return 0
            
            # Count sessions to delete
            count_stmt = select(func.count(QuerySessionModel.id)).where(
                and_(
                    QuerySessionModel.user_id == str(user_id),
                    ~QuerySessionModel.id.in_(keep_ids)
                )
            )
            result = await db.execute(count_stmt)
            delete_count = result.scalar()
            
            # Delete old sessions
            delete_stmt = delete(QuerySessionModel).where(
                and_(
                    QuerySessionModel.user_id == str(user_id),
                    ~QuerySessionModel.id.in_(keep_ids)
                )
            )
            await db.execute(delete_stmt)
            await db.commit()
            
            return delete_count
    
    async def count(self, db: AsyncSession, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count query sessions with optional filtering."""
        stmt = select(func.count(QuerySessionModel.id)).where(
            QuerySessionModel.expires_at > datetime.utcnow()
        )
        
        if filters:
            stmt = self._apply_filters(stmt, filters)
        
        result = await db.execute(stmt)
        return result.scalar()
    
    async def count_active_by_user(self, db: AsyncSession, user_id: UUID) -> int:
        """Count active query sessions for a user."""
        with db.no_autoflush:
            stmt = select(func.count(QuerySessionModel.id)).where(
                and_(
                    QuerySessionModel.user_id == str(user_id),
                    QuerySessionModel.expires_at > datetime.utcnow()
                )
            )
            
            result = await db.execute(stmt)
            return result.scalar()
    
    async def get_session_statistics(
        self, 
        db: AsyncSession, 
        user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Get statistics about query sessions."""
        with db.no_autoflush:
            base_filter = QuerySessionModel.expires_at > datetime.utcnow()
            if user_id:
                base_filter = and_(base_filter, QuerySessionModel.user_id == str(user_id))
            
            # Count active sessions
            active_stmt = select(func.count(QuerySessionModel.id)).where(base_filter)
            active_result = await db.execute(active_stmt)
            active_count = active_result.scalar()
            
            # Count expired sessions
            expired_filter = QuerySessionModel.expires_at <= datetime.utcnow()
            if user_id:
                expired_filter = and_(expired_filter, QuerySessionModel.user_id == str(user_id))
            
            expired_stmt = select(func.count(QuerySessionModel.id)).where(expired_filter)
            expired_result = await db.execute(expired_stmt)
            expired_count = expired_result.scalar()
            
            # Get average session duration (for active sessions)
            # SQLite doesn't have EXTRACT, so we'll use a simpler calculation
            duration_stmt = select(
                func.avg(
                    func.julianday(QuerySessionModel.updated_at) - func.julianday(QuerySessionModel.created_at)
                ) * 24 * 60  # Convert days to minutes
            ).where(base_filter)
            duration_result = await db.execute(duration_stmt)
            avg_duration_minutes = duration_result.scalar() or 0
            
            return {
                "active_sessions": active_count,
                "expired_sessions": expired_count,
                "total_sessions": active_count + expired_count,
                "average_duration_minutes": round(avg_duration_minutes, 2) if avg_duration_minutes else 0
            }
    
    def _apply_filters(self, stmt, filters: Dict[str, Any]):
        """Apply filters to a SQLAlchemy statement."""
        if "user_id" in filters:
            stmt = stmt.where(QuerySessionModel.user_id == str(filters["user_id"]))
        
        if "created_after" in filters:
            stmt = stmt.where(QuerySessionModel.created_at >= filters["created_after"])
        
        if "created_before" in filters:
            stmt = stmt.where(QuerySessionModel.created_at <= filters["created_before"])
        
        if "updated_after" in filters:
            stmt = stmt.where(QuerySessionModel.updated_at >= filters["updated_after"])
        
        if "updated_before" in filters:
            stmt = stmt.where(QuerySessionModel.updated_at <= filters["updated_before"])
        
        if "has_media_references" in filters:
            if filters["has_media_references"]:
                # For SQLite, search within the JSON string
                stmt = stmt.where(
                    and_(
                        QuerySessionModel.session_context.isnot(None),
                        QuerySessionModel.session_context.like('%"media_references"%')
                    )
                )
            else:
                stmt = stmt.where(
                    or_(
                        QuerySessionModel.session_context.is_(None),
                        ~QuerySessionModel.session_context.like('%"media_references"%')
                    )
                )
        
        if "last_query_contains" in filters:
            stmt = stmt.where(
                and_(
                    QuerySessionModel.last_query.isnot(None),
                    QuerySessionModel.last_query.like(f"%{filters['last_query_contains']}%")
                )
            )
        
        return stmt


# Create a singleton instance
test_query_session_repository = QuerySessionModelRepository()