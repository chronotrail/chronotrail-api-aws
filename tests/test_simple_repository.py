"""
Simple test to verify the location visits repository implementation works.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from app.models.schemas import LocationVisitCreate, LocationVisitUpdate
from app.repositories.location_visits import location_visit_repository


def test_repository_exists():
    """Test that the repository exists and has the required methods."""
    # Test that the repository has all the required methods
    assert hasattr(location_visit_repository, "create")
    assert hasattr(location_visit_repository, "get")
    assert hasattr(location_visit_repository, "get_by_user")
    assert hasattr(location_visit_repository, "get_by_user_with_date_range")
    assert hasattr(location_visit_repository, "search_by_names")
    assert hasattr(location_visit_repository, "search_by_address")
    assert hasattr(location_visit_repository, "update_by_id")
    assert hasattr(location_visit_repository, "delete_by_user")
    assert hasattr(location_visit_repository, "count_by_user")
    assert hasattr(location_visit_repository, "get_recent_by_user")
    assert hasattr(location_visit_repository, "get_visits_with_descriptions")
    assert hasattr(location_visit_repository, "get_by_coordinates")


def test_location_visit_create_schema():
    """Test that LocationVisitCreate schema works correctly."""
    visit_data = LocationVisitCreate(
        longitude=Decimal("-122.4194"),
        latitude=Decimal("37.7749"),
        address="San Francisco, CA",
        names=["Golden Gate Park", "Park"],
        visit_time=datetime.utcnow() - timedelta(hours=2),
        duration=120,
        description="Visited the beautiful Golden Gate Park",
    )

    assert visit_data.longitude == Decimal("-122.4194")
    assert visit_data.latitude == Decimal("37.7749")
    assert visit_data.address == "San Francisco, CA"
    assert visit_data.names == ["Golden Gate Park", "Park"]
    assert visit_data.duration == 120
    assert visit_data.description == "Visited the beautiful Golden Gate Park"


def test_location_visit_update_schema():
    """Test that LocationVisitUpdate schema works correctly."""
    update_data = LocationVisitUpdate(
        description="Updated description", names=["Updated Park", "New Name"]
    )

    assert update_data.description == "Updated description"
    assert update_data.names == ["Updated Park", "New Name"]


def test_repository_methods_callable():
    """Test that repository methods are callable (basic smoke test)."""
    # This is a basic smoke test to ensure methods exist and are callable
    # We can't test actual database operations without a database connection

    # Test that methods exist and are callable
    assert callable(location_visit_repository.create)
    assert callable(location_visit_repository.get)
    assert callable(location_visit_repository.get_by_user)
    assert callable(location_visit_repository.get_by_user_with_date_range)
    assert callable(location_visit_repository.search_by_names)
    assert callable(location_visit_repository.search_by_address)
    assert callable(location_visit_repository.update_by_id)
    assert callable(location_visit_repository.delete_by_user)
    assert callable(location_visit_repository.count_by_user)
    assert callable(location_visit_repository.get_recent_by_user)
    assert callable(location_visit_repository.get_visits_with_descriptions)
    assert callable(location_visit_repository.get_by_coordinates)


def test_repository_inheritance():
    """Test that the repository inherits from BaseRepository correctly."""
    from app.repositories.base import BaseRepository

    # Test that the repository is an instance of BaseRepository
    assert isinstance(location_visit_repository, BaseRepository)

    # Test that it has the required model
    assert hasattr(location_visit_repository, "model")

    # Test that the model is the correct type
    from app.models.database import LocationVisit

    assert location_visit_repository.model == LocationVisit
