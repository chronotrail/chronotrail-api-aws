# Implementation Plan

- [x] 1. Set up project structure and core dependencies
  - Create FastAPI project structure with proper directory organization
  - Configure UV package management with pyproject.toml and all AWS SDK dependencies
  - Set up environment configuration for AWS services and database connections
  - _Requirements: 10.1, 10.2_

- [x] 2. Implement database models and migrations
  - [x] 2.1 Create SQLAlchemy models for all database tables
    - Implement User, LocationVisit, TextNote, MediaFile, QuerySession, and DailyUsage models
    - Configure proper relationships and constraints with OAuth fields and subscription tiers
    - Add array field support for names/tags
    - _Requirements: 1.1, 2.1, 7.1, 8.1, 9.1_
  
  - [x] 2.2 Create database migration scripts
    - Write Alembic migrations for all table creation
    - Add proper indexes including GIN indexes for array fields
    - Include sample data for testing
    - _Requirements: 1.1, 2.1, 8.1, 9.1_

- [x] 3. Implement Pydantic data models and validation
  - Create all request/response models (LocationVisit, TextNote, MediaFile, QueryRequest, QueryResponse)
  - Add User, UsageStats, SubscriptionInfo models with OAuth and tier validation
  - Add proper validation rules for coordinates, timestamps, and file types
  - Implement MediaReference and ErrorResponse models
  - _Requirements: 1.2, 2.2, 3.2, 4.2, 7.1, 8.2, 9.2_

- [-] 4. Set up AWS service integrations
  - [x] 4.1 Configure AWS clients and authentication
    - Set up boto3 clients for S3, OpenSearch, Bedrock, Textract, Transcribe, Rekognition
    - Implement proper AWS credential management and error handling
    - Create connection pooling and retry logic
    - _Requirements: 3.1, 4.1, 5.1, 6.1_
  
  - [x] 4.2 Implement S3 file storage service
    - Create file upload/download functionality with proper naming conventions
    - Add file type validation and size limits
    - Implement secure URL generation for media retrieval
    - _Requirements: 3.5, 4.5, 6.2, 6.4_

- [ ] 5. Implement core processing services
  - [x] 5.1 Create OCR and image analysis service
    - Integrate Amazon Textract for text extraction from images
    - Integrate Amazon Rekognition for image content description
    - Handle processing failures with fallback mechanisms
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [x] 5.2 Create speech-to-text transcription service
    - Integrate Amazon Transcribe for voice note processing
    - Handle various audio formats and quality levels
    - Implement error handling for transcription failures
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [x] 5.3 Create embedding and vector storage service
    - Integrate Amazon Bedrock for text embeddings generation
    - Implement OpenSearch document indexing with proper mapping
    - Create vector search functionality with user isolation
    - _Requirements: 2.1, 3.2, 4.2, 5.2_

- [ ] 6. Implement OAuth authentication and usage management
  - [x] 6.1 Create OAuth integration services
    - Implement Google OAuth token verification with Google API client
    - Implement Apple Sign-In token verification with Apple's public keys
    - Create user creation/update logic from OAuth provider data
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [x] 6.2 Implement JWT token management
    - Create JWT token generation with user claims and subscription tier
    - Implement JWT token validation middleware for protected endpoints
    - Add refresh token functionality for token renewal
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [x] 6.3 Create usage tracking and enforcement service
    - Implement daily usage tracking for content submissions and queries
    - Create tier-based limit enforcement with proper error responses
    - Add query date range validation based on subscription tier
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 7. Create database repository layer
  - [x] 7.1 Implement location visits repository
    - Create CRUD operations for location visits with array field support
    - Add date range filtering and coordinate-based queries
    - Implement update functionality for descriptions and names
    - _Requirements: 1.1, 1.3, 8.1, 8.2, 9.1, 9.5_
  
  - [x] 7.2 Implement text notes and media files repositories
    - Create storage and retrieval operations for text notes and media files
    - Add location-based filtering and name/tag searches
    - Implement proper user isolation in all queries
    - _Requirements: 2.1, 2.3, 3.1, 3.5, 4.1, 4.5_
  
  - [x] 7.3 Implement query session management
    - Create session storage and retrieval for conversation context
    - Add session expiration and cleanup functionality
    - Implement context tracking for media references
    - _Requirements: 5.1, 5.2, 6.1_

- [ ] 8. Implement authentication API endpoints
  - [x] 8.1 Create OAuth authentication endpoints
    - Implement POST /api/v1/auth/google for Google OAuth sign-in
    - Implement POST /api/v1/auth/apple for Apple Sign-In
    - Implement POST /api/v1/auth/refresh for JWT token refresh
    - Implement GET /api/v1/auth/me for user profile retrieval
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [x] 8.2 Create usage and subscription endpoints
    - Implement GET /api/v1/usage for current usage statistics
    - Implement GET /api/v1/subscription for subscription details
    - Add usage limit validation middleware for protected endpoints
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
    
    **Implementation Summary:**
    - **GET /api/v1/usage** - Returns daily usage statistics (text notes, media files, queries) with tier-based limits
    - **GET /api/v1/subscription** - Returns subscription details including tier, expiration, and all limits
    - **Usage validation middleware** - Provides dependency functions for content, query, file size, and date range validation
    - **Files created:** `app/api/v1/endpoints/usage.py`, `app/middleware/usage_validation.py`, `tests/test_usage_endpoints.py`
    - **Files modified:** `app/api/v1/api.py` (added usage router)
    - **Testing:** 39 tests pass (13 new endpoint tests + 26 existing service tests)
    - **Features:** Authentication required, subscription tier aware, comprehensive error handling, extensible middleware design

- [ ] 9. Implement API endpoints for data submission
  - [x] 9.1 Create location visits endpoints
    - Implement POST /api/v1/locations for creating location visits
    - Implement GET /api/v1/locations with date filtering
    - Implement PUT /api/v1/locations/{visit_id} for updates
    - _Requirements: 1.1, 1.2, 1.3, 8.1, 8.2, 8.3, 8.4, 9.1, 9.2, 9.3, 9.4, 9.5_
    
    **Implementation Summary:**
    - **POST /api/v1/locations** - Create location visits with coordinates, address, names, visit time, duration, and description validation
    - **GET /api/v1/locations** - Retrieve location visits with date range filtering, pagination, and subscription tier-based historical data limits
    - **PUT /api/v1/locations/{visit_id}** - Update location visit descriptions and names/tags with user ownership validation
    - **GET /api/v1/locations/{visit_id}** - Retrieve specific location visit by ID with user access control
    - **Files created:** `app/api/v1/endpoints/locations.py`, `tests/test_location_visits_endpoints.py`
    - **Files modified:** `app/api/v1/api.py` (added locations router)
    - **Testing:** 15 comprehensive tests covering all endpoints, validation, authentication, and error scenarios
    - **Features:** JWT authentication required, subscription validity checks, coordinate validation, future date prevention, user data isolation, comprehensive error handling
  
  - [x] 9.2 Create text notes endpoint
    - Implement POST /api/v1/notes for text note submission
    - Add location data processing and vector embedding generation
    - Include proper validation and error handling
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [x] 9.3 Create photo upload endpoint
    - Implement POST /api/v1/photos with file upload handling
    - Add OCR processing and image analysis integration
    - Store processed content in vector database with location context
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [x] 9.4 Create voice upload endpoint
    - Implement POST /api/v1/voice with audio file handling
    - Add transcription processing and vector embedding
    - Handle various audio formats with proper error responses
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 10. Implement natural language query system
  - [x] 10.1 Create query processing engine
    - Implement vector similarity search with user isolation
    - Add structured data filtering for location and time-based queries
    - Create result ranking and relevance scoring
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [x] 10.2 Create LLM integration for response generation
    - Integrate Amazon Bedrock for natural language response generation
    - Implement context assembly from search results
    - Add media reference identification and description generation
    - _Requirements: 5.1, 5.2, 6.1, 6.2_
  
  - [x] 10.3 Implement query API endpoint
    - Create POST /api/v1/query endpoint with session management
    - Add conversation context handling and media reference tracking
    - Implement proper error handling for query failures
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 6.1_

- [x] 11. Implement media retrieval system
  - Create GET /api/v1/media/{media_id} endpoint for file access
  - Add proper authorization checks for media file access
  - Implement secure URL generation and file streaming
  - _Requirements: 6.2, 6.3, 6.4_
 
- [x] 12. Add comprehensive error handling and logging
  - Implement structured error responses for all endpoints
  - Add comprehensive logging for debugging and monitoring
  - Create proper exception handling for AWS service failures
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 13. Create comprehensive test suite
  - [x] 13.1 Write unit tests for all services and repositories
    - Test data models, validation, and business logic
    - Mock AWS services for isolated testing
    - Test error handling and edge cases
    - _Requirements: All requirements_
  
  - [x] 13.2 Write integration tests for API endpoints
    - Test complete request/response cycles for all endpoints
    - Test file upload and processing workflows
    - Test authentication and authorization flows
    - _Requirements: All requirements_
  
  - [x] 13.3 Write end-to-end tests for user workflows
    - Test complete user journeys from data submission to querying
    - Test conversational media retrieval flows
    - Test location tagging and search functionality
    - _Requirements: All requirements_

- [x] 14. Set up deployment configuration
  - Create Docker configuration for ECS Fargate deployment
  - Set up AWS infrastructure as code (CloudFormation/CDK)
  - Configure environment-specific settings and secrets management
  - _Requirements: 10.1, 10.2, 10.3, 10.4_   
 **Implementation Summary:**
    - **POST /api/v1/query** - Natural language query endpoint with session management and conversation context
    - **LLM Integration** - Amazon Bedrock integration for natural language response generation
    - **Context Assembly** - Intelligent assembly of search results for LLM context
    - **Media Reference Tracking** - Identification and tracking of media references in responses
    - **Files created:** `app/aws/llm.py`, `app/api/v1/endpoints/query.py`, `tests/test_llm.py`, `tests/test_query_endpoints.py`
    - **Files modified:** `app/services/query.py`, `app/api/v1/api.py`, `app/middleware/usage_validation.py`
    - **Testing:** Comprehensive tests for LLM service and query API endpoint
    - **Features:** Subscription tier awareness, fallback response generation, conversation context management, media reference tracking