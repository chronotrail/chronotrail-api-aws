"""
Tests for location visits API endpoints.
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from unittest.mock import AsyncMock, patch

from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.services.auth import get_current_active_user
from app.db.database import get_async_db
from app.models.schemas import LocationVisitCreate, LocationVisitUpdate, User
from app.models.database import LocationVisit as DBLocationVisit


class TestLocationVisitsEndpoints:
    """Test cases for location visits endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return AsyncMock(spec=AsyncSession)
    
    @pytest.fixture
    def sample_user(self):
        """Create a sample user for testing."""
        return User(
            id=uuid4(),
            email="test@example.com",
            oauth_provider="google",
            oauth_subject="google_123",
            display_name="Test User",
            subscription_tier="free",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_location_visit_create(self):
        """Create sample location visit creation data."""
        return LocationVisitCreate(
            longitude=Decimal("-122.4194"),
            latitude=Decimal("37.7749"),
            address="San Francisco, CA, USA",
            names=["Golden Gate Park", "Park"],
            visit_time=datetime.utcnow() - timedelta(hours=2),
            duration=120,
            description="Visited the beautiful Golden Gate Park"
        )
    
    @pytest.fixture
    def sample_db_location_visit(self, sample_user, sample_location_visit_create):
        """Create a sample database location visit."""
        return DBLocationVisit(
            id=uuid4(),
            user_id=sample_user.id,
            longitude=sample_location_visit_create.longitude,
            latitude=sample_location_visit_create.latitude,
            address=sample_location_visit_create.address,
            names=sample_location_visit_create.names,
            visit_time=sample_location_visit_create.visit_time,
            duration=sample_location_visit_create.duration,
            description=sample_location_visit_create.description,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    def test_create_location_visit_success(
        self, 
        client, 
        mock_db,
        sample_user,
        sample_location_visit_create,
        sample_db_location_visit
    ):
        """Test successful location visit creation."""
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user), \
             patch("app.repositories.location_visits.location_visit_repository.create", 
                   return_value=sample_db_location_visit) as mock_create:
            
            response = client.post(
                "/api/v1/locations",
                json={
                    "longitude": float(sample_location_visit_create.longitude),
                    "latitude": float(sample_location_visit_create.latitude),
                    "address": sample_location_visit_create.address,
                    "names": sample_location_visit_create.names,
                    "visit_time": sample_location_visit_create.visit_time.isoformat(),
                    "duration": sample_location_visit_create.duration,
                    "description": sample_location_visit_create.description
                },
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            
            assert data["id"] == str(sample_db_location_visit.id)
            assert data["user_id"] == str(sample_user.id)
            assert float(data["longitude"]) == float(sample_location_visit_create.longitude)
            assert float(data["latitude"]) == float(sample_location_visit_create.latitude)
            assert data["address"] == sample_location_visit_create.address
            assert data["names"] == sample_location_visit_create.names
            assert data["duration"] == sample_location_visit_create.duration
            assert data["description"] == sample_location_visit_create.description
            
            # Verify repository was called correctly
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args.kwargs["user_id"] == sample_user.id
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_create_location_visit_invalid_coordinates(
        self, 
        client, 
        mock_db,
        sample_user
    ):
        """Test location visit creation with invalid coordinates."""
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user):
            
            response = client.post(
                "/api/v1/locations",
                json={
                    "longitude": 200.0,  # Invalid longitude
                    "latitude": 37.7749,
                    "address": "San Francisco, CA, USA",
                    "visit_time": datetime.utcnow().isoformat()
                },
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_create_location_visit_future_time(
        self, 
        client, 
        mock_db,
        sample_user
    ):
        """Test location visit creation with future visit time."""
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user):
            
            future_time = datetime.utcnow() + timedelta(hours=1)
            
            response = client.post(
                "/api/v1/locations",
                json={
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "San Francisco, CA, USA",
                    "visit_time": future_time.isoformat()
                },
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_get_location_visits_success(
        self, 
        client, 
        mock_db,
        sample_user,
        sample_db_location_visit
    ):
        """Test successful retrieval of location visits."""
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user), \
             patch("app.services.usage.UsageService.validate_query_date_range", return_value=True), \
             patch("app.repositories.location_visits.location_visit_repository.get_by_user_with_date_range", 
                   return_value=[sample_db_location_visit]) as mock_get, \
             patch("app.repositories.location_visits.location_visit_repository.count_by_user", 
                   return_value=1) as mock_count:
            
            response = client.get(
                "/api/v1/locations",
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["total"] == 1
            assert data["page"] == 1
            assert data["page_size"] == 20
            assert len(data["visits"]) == 1
            
            visit = data["visits"][0]
            assert visit["id"] == str(sample_db_location_visit.id)
            assert visit["user_id"] == str(sample_user.id)
            
            # Verify repository calls
            mock_get.assert_called_once()
            mock_count.assert_called_once()
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_get_location_visits_with_date_filter(
        self, 
        client, 
        mock_db,
        sample_user,
        sample_db_location_visit
    ):
        """Test location visits retrieval with date filtering."""
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()
        
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user), \
             patch("app.services.usage.UsageService.validate_query_date_range", return_value=True), \
             patch("app.repositories.location_visits.location_visit_repository.get_by_user_with_date_range", 
                   return_value=[sample_db_location_visit]) as mock_get, \
             patch("app.repositories.location_visits.location_visit_repository.count_by_user", 
                   return_value=1):
            
            response = client.get(
                "/api/v1/locations",
                params={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            
            # Verify repository was called with date filters
            call_args = mock_get.call_args
            assert call_args.kwargs["start_date"] == start_date
            assert call_args.kwargs["end_date"] == end_date
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_get_location_visits_invalid_date_range(
        self, 
        client, 
        mock_db,
        sample_user
    ):
        """Test location visits retrieval with invalid date range."""
        start_date = datetime.utcnow()
        end_date = datetime.utcnow() - timedelta(days=1)  # End before start
        
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user):
            
            response = client.get(
                "/api/v1/locations",
                params={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_get_location_visits_with_pagination(
        self, 
        client, 
        mock_db,
        sample_user,
        sample_db_location_visit
    ):
        """Test location visits retrieval with pagination."""
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user), \
             patch("app.services.usage.UsageService.validate_query_date_range", return_value=True), \
             patch("app.repositories.location_visits.location_visit_repository.get_by_user_with_date_range", 
                   return_value=[sample_db_location_visit]) as mock_get, \
             patch("app.repositories.location_visits.location_visit_repository.count_by_user", 
                   return_value=50):
            
            response = client.get(
                "/api/v1/locations",
                params={"page": 2, "page_size": 10},
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["page"] == 2
            assert data["page_size"] == 10
            assert data["total"] == 50
            
            # Verify pagination offset calculation
            call_args = mock_get.call_args
            assert call_args.kwargs["skip"] == 10  # (page - 1) * page_size
            assert call_args.kwargs["limit"] == 10
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_update_location_visit_success(
        self, 
        client, 
        mock_db,
        sample_user,
        sample_db_location_visit
    ):
        """Test successful location visit update."""
        visit_id = sample_db_location_visit.id
        update_data = LocationVisitUpdate(
            description="Updated description",
            names=["Updated Park", "New Name"]
        )
        
        # Create updated visit object
        updated_visit = DBLocationVisit(
            id=visit_id,
            user_id=sample_user.id,
            longitude=sample_db_location_visit.longitude,
            latitude=sample_db_location_visit.latitude,
            address=sample_db_location_visit.address,
            names=update_data.names,
            visit_time=sample_db_location_visit.visit_time,
            duration=sample_db_location_visit.duration,
            description=update_data.description,
            created_at=sample_db_location_visit.created_at,
            updated_at=datetime.utcnow()
        )
        
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user), \
             patch("app.repositories.location_visits.location_visit_repository.update_by_id", 
                   return_value=updated_visit) as mock_update:
            
            response = client.put(
                f"/api/v1/locations/{visit_id}",
                json={
                    "description": update_data.description,
                    "names": update_data.names
                },
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["id"] == str(visit_id)
            assert data["description"] == update_data.description
            assert data["names"] == update_data.names
            
            # Verify repository was called correctly
            mock_update.assert_called_once()
            call_args = mock_update.call_args
            assert call_args.kwargs["user_id"] == sample_user.id
            assert call_args.kwargs["visit_id"] == visit_id
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_update_location_visit_not_found(
        self, 
        client, 
        mock_db,
        sample_user
    ):
        """Test updating non-existent location visit."""
        visit_id = uuid4()
        
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user), \
             patch("app.repositories.location_visits.location_visit_repository.update_by_id", 
                   return_value=None):
            
            response = client.put(
                f"/api/v1/locations/{visit_id}",
                json={"description": "Updated description"},
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_update_location_visit_invalid_names(
        self, 
        client, 
        mock_db,
        sample_user
    ):
        """Test updating location visit with invalid names."""
        visit_id = uuid4()
        
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user):
            
            response = client.put(
                f"/api/v1/locations/{visit_id}",
                json={
                    "names": ["", "  ", "valid_name"]  # Empty and whitespace names
                },
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_get_location_visit_by_id_success(
        self, 
        client, 
        mock_db,
        sample_user,
        sample_db_location_visit
    ):
        """Test successful retrieval of location visit by ID."""
        visit_id = sample_db_location_visit.id
        
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user), \
             patch("app.repositories.location_visits.location_visit_repository.get_by_user", 
                   return_value=sample_db_location_visit) as mock_get:
            
            response = client.get(
                f"/api/v1/locations/{visit_id}",
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["id"] == str(visit_id)
            assert data["user_id"] == str(sample_user.id)
            assert data["address"] == sample_db_location_visit.address
            
            # Verify repository was called correctly
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args.kwargs["user_id"] == sample_user.id
            assert call_args.kwargs["visit_id"] == visit_id
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_get_location_visit_by_id_not_found(
        self, 
        client, 
        mock_db,
        sample_user
    ):
        """Test retrieval of non-existent location visit."""
        visit_id = uuid4()
        
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user), \
             patch("app.repositories.location_visits.location_visit_repository.get_by_user", 
                   return_value=None):
            
            response = client.get(
                f"/api/v1/locations/{visit_id}",
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_location_visits_require_authentication(self, client):
        """Test that location visits endpoints require authentication."""
        # Test POST endpoint
        response = client.post(
            "/api/v1/locations",
            json={
                "longitude": -122.4194,
                "latitude": 37.7749,
                "visit_time": datetime.utcnow().isoformat()
            }
        )
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
        
        # Test GET endpoint
        response = client.get("/api/v1/locations")
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
        
        # Test PUT endpoint
        visit_id = uuid4()
        response = client.put(
            f"/api/v1/locations/{visit_id}",
            json={"description": "test"}
        )
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
        
        # Test GET by ID endpoint
        response = client.get(f"/api/v1/locations/{visit_id}")
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
    
    def test_subscription_validity_check(
        self, 
        client, 
        mock_db,
        sample_user
    ):
        """Test that subscription validity is checked for all endpoints."""
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        from fastapi import HTTPException
        
        with patch("app.services.usage.UsageService.check_subscription_validity", 
                   side_effect=HTTPException(status_code=403, detail="Subscription expired")) as mock_check:
            
            # Test POST endpoint
            response = client.post(
                "/api/v1/locations",
                json={
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "visit_time": datetime.utcnow().isoformat()
                },
                headers={"Authorization": "Bearer fake_token"}
            )
            assert response.status_code == status.HTTP_403_FORBIDDEN
            
            # Verify subscription check was called
            mock_check.assert_called()
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_query_date_range_validation(
        self, 
        client, 
        mock_db,
        sample_user
    ):
        """Test query date range validation for free tier users."""
        old_date = datetime.utcnow() - timedelta(days=365)  # 1 year ago
        
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        from fastapi import HTTPException
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user), \
             patch("app.services.usage.UsageService.validate_query_date_range", 
                   side_effect=HTTPException(status_code=403, detail="Date range exceeds limits")) as mock_validate:
            
            response = client.get(
                "/api/v1/locations",
                params={"start_date": old_date.isoformat()},
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_403_FORBIDDEN
            
            # Verify date range validation was called
            mock_validate.assert_called_once_with(sample_user, old_date)
        
        # Clean up
        app.dependency_overrides.clear()