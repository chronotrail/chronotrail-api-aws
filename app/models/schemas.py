"""
Pydantic models for request/response validation and serialization.
"""
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator, EmailStr, ConfigDict


class OAuthProvider(str, Enum):
    """Supported OAuth providers."""
    GOOGLE = "google"
    APPLE = "apple"


class SubscriptionTier(str, Enum):
    """Available subscription tiers."""
    FREE = "free"
    PREMIUM = "premium"
    PRO = "pro"


class FileType(str, Enum):
    """Supported media file types."""
    PHOTO = "photo"
    VOICE = "voice"
    
    
class AllowedPhotoExtensions(str, Enum):
    """Allowed photo file extensions."""
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    HEIC = "heic"
    WEBP = "webp"
    
    
class AllowedVoiceExtensions(str, Enum):
    """Allowed voice file extensions."""
    MP3 = "mp3"
    WAV = "wav"
    M4A = "m4a"
    AAC = "aac"
    OGG = "ogg"


class ContentType(str, Enum):
    """Content types for vector database storage."""
    NOTE = "note"
    LOCATION_DESC = "location_desc"
    IMAGE_TEXT = "image_text"
    IMAGE_DESC = "image_desc"
    VOICE_TRANSCRIPT = "voice_transcript"


# Base models for common fields
class TimestampMixin(BaseModel):
    """Mixin for models with timestamp fields."""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class LocationMixin(BaseModel):
    """Mixin for models with optional location data."""
    longitude: Optional[Decimal] = Field(None, ge=-180, le=180, decimal_places=8)
    latitude: Optional[Decimal] = Field(None, ge=-90, le=90, decimal_places=8)
    address: Optional[str] = Field(None, max_length=1000)
    names: Optional[List[str]] = Field(None, max_length=10)

    @field_validator('names')
    @classmethod
    def validate_names(cls, v):
        if v is not None:
            # Filter out empty strings and limit length
            filtered = [name.strip() for name in v if name and name.strip()]
            if len(filtered) != len(v):
                raise ValueError("Names cannot be empty or contain only whitespace")
            for name in filtered:
                if len(name) > 100:
                    raise ValueError("Each name must be 100 characters or less")
            return filtered[:10]  # Limit to 10 names
        return v

    @model_validator(mode='before')
    @classmethod
    def validate_location_consistency(cls, values):
        """Ensure longitude and latitude are both provided or both None."""
        if isinstance(values, dict):
            longitude = values.get('longitude')
            latitude = values.get('latitude')
            
            if (longitude is None) != (latitude is None):
                raise ValueError("Both longitude and latitude must be provided together or both omitted")
        
        return values


# User-related models
class UserBase(BaseModel):
    """Base user model with common fields."""
    email: EmailStr
    display_name: Optional[str] = Field(None, max_length=255)
    profile_picture_url: Optional[str] = Field(None, max_length=2000)


class UserCreate(UserBase):
    """Model for creating a new user."""
    oauth_provider: OAuthProvider
    oauth_subject: str = Field(..., max_length=255)
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    subscription_expires_at: Optional[datetime] = None


class UserUpdate(BaseModel):
    """Model for updating user information."""
    display_name: Optional[str] = Field(None, max_length=255)
    profile_picture_url: Optional[str] = Field(None, max_length=2000)
    subscription_tier: Optional[SubscriptionTier] = None
    subscription_expires_at: Optional[datetime] = None


class User(UserBase, TimestampMixin):
    """Complete user model for responses."""
    id: UUID
    oauth_provider: OAuthProvider
    oauth_subject: str
    subscription_tier: SubscriptionTier
    subscription_expires_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# Usage tracking models
class UsageStats(BaseModel):
    """Daily usage statistics for a user."""
    date: date
    text_notes_count: int = Field(ge=0)
    media_files_count: int = Field(ge=0)
    queries_count: int = Field(ge=0)
    daily_limits: Dict[str, int]

    model_config = ConfigDict(from_attributes=True)


class SubscriptionInfo(BaseModel):
    """Subscription information and limits."""
    tier: SubscriptionTier
    expires_at: Optional[datetime] = None
    daily_limits: Dict[str, int]
    query_history_months: int
    max_file_size_mb: int
    max_storage_mb: int

    model_config = ConfigDict(from_attributes=True)


# Location visit models
class LocationVisitBase(LocationMixin):
    """Base location visit model."""
    visit_time: datetime
    duration: Optional[int] = Field(None, ge=0, description="Duration in minutes")
    description: Optional[str] = Field(None, max_length=5000)

    @field_validator('visit_time')
    @classmethod
    def validate_visit_time(cls, v):
        """Ensure visit time is not in the future."""
        if v > datetime.utcnow():
            raise ValueError("Visit time cannot be in the future")
        return v


class LocationVisitCreate(LocationVisitBase):
    """Model for creating a location visit."""
    pass


class LocationVisitUpdate(BaseModel):
    """Model for updating a location visit."""
    description: Optional[str] = Field(None, max_length=5000)
    names: Optional[List[str]] = Field(None, max_length=10)

    @field_validator('names')
    @classmethod
    def validate_names(cls, v):
        if v is not None:
            filtered = [name.strip() for name in v if name and name.strip()]
            if len(filtered) != len(v):
                raise ValueError("Names cannot be empty or contain only whitespace")
            for name in filtered:
                if len(name) > 100:
                    raise ValueError("Each name must be 100 characters or less")
            return filtered[:10]
        return v


class LocationVisit(LocationVisitBase, TimestampMixin):
    """Complete location visit model for responses."""
    id: UUID
    user_id: UUID

    model_config = ConfigDict(from_attributes=True)


# Text note models
class TextNoteBase(LocationMixin):
    """Base text note model."""
    text_content: str = Field(..., min_length=1, max_length=10000)
    timestamp: datetime

    @field_validator('text_content')
    @classmethod
    def validate_text_content(cls, v):
        """Ensure text content is not just whitespace."""
        if not v.strip():
            raise ValueError("Text content cannot be empty or contain only whitespace")
        return v.strip()

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        """Ensure timestamp is not in the future."""
        if v > datetime.utcnow():
            raise ValueError("Timestamp cannot be in the future")
        return v


class TextNoteCreate(TextNoteBase):
    """Model for creating a text note."""
    pass


class TextNote(TextNoteBase, TimestampMixin):
    """Complete text note model for responses."""
    id: UUID
    user_id: UUID

    model_config = ConfigDict(from_attributes=True)


# Media file models
class MediaFileBase(LocationMixin):
    """Base media file model."""
    file_type: FileType
    original_filename: Optional[str] = Field(None, max_length=255)
    timestamp: datetime

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        """Ensure timestamp is not in the future."""
        if v > datetime.utcnow():
            raise ValueError("Timestamp cannot be in the future")
        return v
    
    @field_validator('original_filename')
    @classmethod
    def validate_filename(cls, v):
        """Validate filename if provided."""
        if v is not None:
            if not v.strip():
                raise ValueError("Filename cannot be empty or contain only whitespace")
            # Check for invalid characters in filename
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            if any(char in v for char in invalid_chars):
                raise ValueError(f"Filename contains invalid characters: {', '.join(invalid_chars)}")
        return v


class MediaFileCreate(MediaFileBase):
    """Model for creating a media file record."""
    
    @model_validator(mode='after')
    def validate_file_type_consistency(self):
        """Validate file type and filename extension consistency."""
        if self.original_filename and self.file_type:
            extension = self.original_filename.split('.')[-1].lower() if '.' in self.original_filename else None
            
            if self.file_type == FileType.PHOTO and extension:
                try:
                    AllowedPhotoExtensions(extension)
                except ValueError:
                    allowed = [e.value for e in AllowedPhotoExtensions]
                    raise ValueError(f"Invalid photo file extension. Allowed extensions: {', '.join(allowed)}")
            
            elif self.file_type == FileType.VOICE and extension:
                try:
                    AllowedVoiceExtensions(extension)
                except ValueError:
                    allowed = [e.value for e in AllowedVoiceExtensions]
                    raise ValueError(f"Invalid voice file extension. Allowed extensions: {', '.join(allowed)}")
        
        return self


class MediaFile(MediaFileBase, TimestampMixin):
    """Complete media file model for responses."""
    id: UUID
    user_id: UUID
    file_path: str
    file_size: Optional[int] = Field(None, ge=0)

    model_config = ConfigDict(from_attributes=True)


# Query-related models
class QueryRequest(BaseModel):
    """Model for natural language query requests."""
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, max_length=36)

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or contain only whitespace")
        return v.strip()


class MediaReference(BaseModel):
    """Reference to a media file in query responses."""
    media_id: UUID
    media_type: FileType
    description: str = Field(..., min_length=1, max_length=500)
    timestamp: datetime
    location: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class QueryResponse(BaseModel):
    """Model for natural language query responses."""
    answer: str = Field(..., min_length=1)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    media_references: Optional[List[MediaReference]] = None
    session_id: str = Field(..., max_length=36)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


# Error handling models
class ErrorResponse(BaseModel):
    """Standardized error response model."""
    error: str = Field(..., max_length=100)
    message: str = Field(..., max_length=500)
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


# Request/Response wrapper models
class LocationVisitsResponse(BaseModel):
    """Response model for location visits list."""
    visits: List[LocationVisit]
    total: int = Field(ge=0)
    page: int = Field(ge=1, default=1)
    page_size: int = Field(ge=1, le=100, default=20)

    model_config = ConfigDict(from_attributes=True)


class DateRangeFilter(BaseModel):
    """Model for date range filtering."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    @model_validator(mode='after')
    def validate_date_range(self):
        """Ensure start_date is before end_date."""
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


# File upload models
class FileUploadResponse(BaseModel):
    """Response model for file uploads."""
    file_id: UUID
    message: str
    processing_status: str = Field(default="pending", pattern="^(pending|processing|completed|failed)$")
    error_details: Optional[str] = None
    file_type: FileType
    file_size: Optional[int] = Field(None, ge=0)

    model_config = ConfigDict(from_attributes=True)


# Authentication models
class TokenResponse(BaseModel):
    """Response model for authentication tokens."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class RefreshTokenRequest(BaseModel):
    """Request model for token refresh."""
    refresh_token: str = Field(..., min_length=1)


# Pagination models
class PaginationParams(BaseModel):
    """Model for pagination parameters."""
    page: int = Field(ge=1, default=1)
    page_size: int = Field(ge=1, le=100, default=20)

    model_config = ConfigDict(from_attributes=True)

# File size validation models
class FileSizeLimits(BaseModel):
    """File size limits based on subscription tier."""
    photo_max_size_mb: int = Field(ge=1)
    voice_max_size_mb: int = Field(ge=1)
    total_storage_mb: int = Field(ge=1)
    
    model_config = ConfigDict(from_attributes=True)


# Vector search models
class VectorSearchResult(BaseModel):
    """Model for vector search results."""
    content_id: str
    content_type: ContentType
    content_text: str
    timestamp: datetime
    score: float = Field(ge=0, le=1)
    source_id: Optional[UUID] = None
    location: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


# Processing result models
class ProcessedContent(BaseModel):
    """Base model for processed content."""
    original_id: UUID
    processed_text: str
    content_type: ContentType
    processing_status: str = Field(pattern="^(completed|partial|failed)$")
    error_details: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class ProcessedPhoto(ProcessedContent):
    """Model for processed photo content."""
    extracted_text: Optional[str] = None
    image_description: Optional[str] = None
    detected_objects: Optional[List[str]] = None
    
    model_config = ConfigDict(from_attributes=True)


class ProcessedVoice(ProcessedContent):
    """Model for processed voice content."""
    transcript: str
    confidence_score: float = Field(ge=0, le=1)
    duration_seconds: Optional[float] = Field(None, ge=0)
    
    model_config = ConfigDict(from_attributes=True)


# Query session models
class QuerySession(BaseModel):
    """Model for query session data."""
    id: UUID
    user_id: UUID
    session_context: Optional[Dict[str, Any]] = None
    last_query: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class QuerySessionCreate(BaseModel):
    """Model for creating a new query session."""
    user_id: UUID
    session_context: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class QuerySessionUpdate(BaseModel):
    """Model for updating a query session."""
    session_context: Optional[Dict[str, Any]] = None
    last_query: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


# OAuth request models
class GoogleOAuthRequest(BaseModel):
    """Request model for Google OAuth authentication."""
    access_token: str = Field(..., min_length=1, description="Google OAuth access token")


class AppleOAuthRequest(BaseModel):
    """Request model for Apple Sign-In authentication."""
    identity_token: str = Field(..., min_length=1, description="Apple Sign-In identity token")


# OAuth token validation models
class OAuthTokenInfo(BaseModel):
    """Model for OAuth token information."""
    provider: OAuthProvider
    subject: str
    email: EmailStr
    name: Optional[str] = None
    picture_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class GoogleOAuthToken(BaseModel):
    """Model for Google OAuth token validation."""
    iss: str = Field(..., pattern="^https://accounts\\.google\\.com$")
    sub: str
    email: EmailStr
    email_verified: bool = True
    name: Optional[str] = None
    picture: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    exp: int
    
    @field_validator('email_verified')
    @classmethod
    def validate_email_verified(cls, v):
        """Ensure email is verified."""
        if not v:
            raise ValueError("Email must be verified")
        return v
    
    model_config = ConfigDict(extra='allow')


class AppleOAuthToken(BaseModel):
    """Model for Apple OAuth token validation."""
    iss: str = Field(..., pattern="^https://appleid\\.apple\\.com$")
    sub: str
    email: Optional[EmailStr] = None
    email_verified: Optional[bool] = None
    is_private_email: Optional[bool] = None
    exp: int
    
    model_config = ConfigDict(extra='allow')