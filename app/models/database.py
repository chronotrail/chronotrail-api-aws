"""
SQLAlchemy database models for ChronoTrail API
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    ARRAY,
    DECIMAL,
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base


class User(Base):
    """User model for storing user account information with OAuth integration."""

    __tablename__ = "users"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    oauth_provider = Column(String(50), nullable=False)  # 'google', 'apple'
    oauth_subject = Column(String(255), nullable=False)  # Provider's user ID
    display_name = Column(String(255), nullable=True)
    profile_picture_url = Column(Text, nullable=True)
    subscription_tier = Column(
        String(50), nullable=False, default="free"
    )  # 'free', 'premium', 'pro'
    subscription_expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now()
    )

    # Relationships
    location_visits = relationship(
        "LocationVisit", back_populates="user", cascade="all, delete-orphan"
    )
    text_notes = relationship(
        "TextNote", back_populates="user", cascade="all, delete-orphan"
    )
    media_files = relationship(
        "MediaFile", back_populates="user", cascade="all, delete-orphan"
    )
    query_sessions = relationship(
        "QuerySession", back_populates="user", cascade="all, delete-orphan"
    )
    daily_usage = relationship(
        "DailyUsage", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return (
            f"<User(id={self.id}, email={self.email}, tier={self.subscription_tier})>"
        )


class DailyUsage(Base):
    """Daily usage tracking for subscription tier enforcement."""

    __tablename__ = "daily_usage"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    usage_date = Column(Date, nullable=False)
    text_notes_count = Column(Integer, nullable=False, default=0)
    media_files_count = Column(Integer, nullable=False, default=0)
    queries_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now()
    )

    # Relationships
    user = relationship("User", back_populates="daily_usage")

    def __repr__(self):
        return f"<DailyUsage(user_id={self.user_id}, date={self.usage_date})>"


class LocationVisit(Base):
    """Location visit model for storing user location data with array support for names."""

    __tablename__ = "location_visits"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    longitude = Column(DECIMAL(10, 8), nullable=False)
    latitude = Column(DECIMAL(11, 8), nullable=False)
    address = Column(Text, nullable=True)
    names = Column(ARRAY(Text), nullable=True)  # Array of names/tags for the location
    visit_time = Column(DateTime, nullable=False)
    duration = Column(Integer, nullable=True)  # Duration in minutes
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now()
    )

    # Relationships
    user = relationship("User", back_populates="location_visits")

    def __repr__(self):
        return f"<LocationVisit(id={self.id}, user_id={self.user_id}, visit_time={self.visit_time})>"


class TextNote(Base):
    """Text note model for storing user text content with optional location data."""

    __tablename__ = "text_notes"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    text_content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    longitude = Column(DECIMAL(10, 8), nullable=True)  # Optional location data
    latitude = Column(DECIMAL(11, 8), nullable=True)  # Optional location data
    address = Column(Text, nullable=True)  # Optional location data
    names = Column(ARRAY(Text), nullable=True)  # Array of names/tags for the location
    created_at = Column(DateTime, nullable=False, default=func.now())

    # Relationships
    user = relationship("User", back_populates="text_notes")

    def __repr__(self):
        return f"<TextNote(id={self.id}, user_id={self.user_id}, timestamp={self.timestamp})>"


class MediaFile(Base):
    """Media file model for storing photo and voice file metadata with location data."""

    __tablename__ = "media_files"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    file_type = Column(String(50), nullable=False)  # 'photo', 'voice'
    file_path = Column(String(500), nullable=False)
    original_filename = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=True)
    timestamp = Column(DateTime, nullable=False)
    longitude = Column(DECIMAL(10, 8), nullable=True)  # Optional location data
    latitude = Column(DECIMAL(11, 8), nullable=True)  # Optional location data
    address = Column(Text, nullable=True)  # Optional location data
    names = Column(ARRAY(Text), nullable=True)  # Array of names/tags for the location
    created_at = Column(DateTime, nullable=False, default=func.now())

    # Relationships
    user = relationship("User", back_populates="media_files")

    def __repr__(self):
        return f"<MediaFile(id={self.id}, user_id={self.user_id}, file_type={self.file_type})>"


class QuerySession(Base):
    """Query session model for storing conversation context and media references."""

    __tablename__ = "query_sessions"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_context = Column(
        JSON, nullable=True
    )  # Stores conversation context and referenced media
    last_query = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now()
    )
    expires_at = Column(
        DateTime,
        nullable=False,
        default=lambda: func.now() + func.make_interval(hours=1),
    )

    # Relationships
    user = relationship("User", back_populates="query_sessions")

    def __repr__(self):
        return f"<QuerySession(id={self.id}, user_id={self.user_id}, expires_at={self.expires_at})>"
