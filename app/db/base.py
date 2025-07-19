"""
Database base class for SQLAlchemy models.
"""
from sqlalchemy.ext.declarative import declarative_base

# Create Base class for models
Base = declarative_base()