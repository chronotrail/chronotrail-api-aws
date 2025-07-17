"""Initial migration with all tables and indexes

Revision ID: bfbe8f3086c8
Revises: 
Create Date: 2025-07-16 21:04:11.136367

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'bfbe8f3086c8'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('oauth_provider', sa.String(length=50), nullable=False),
        sa.Column('oauth_subject', sa.String(length=255), nullable=False),
        sa.Column('display_name', sa.String(length=255), nullable=True),
        sa.Column('profile_picture_url', sa.Text(), nullable=True),
        sa.Column('subscription_tier', sa.String(length=50), nullable=False),
        sa.Column('subscription_expires_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('oauth_provider', 'oauth_subject', name='uq_users_oauth_provider_subject')
    )
    op.create_index('idx_users_email', 'users', ['email'], unique=False)
    op.create_index('idx_users_oauth', 'users', ['oauth_provider', 'oauth_subject'], unique=False)
    op.create_index('idx_users_subscription', 'users', ['subscription_tier', 'subscription_expires_at'], unique=False)

    # Create daily_usage table
    op.create_table('daily_usage',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('usage_date', sa.Date(), nullable=False),
        sa.Column('text_notes_count', sa.Integer(), nullable=False),
        sa.Column('media_files_count', sa.Integer(), nullable=False),
        sa.Column('queries_count', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'usage_date', name='uq_daily_usage_user_date')
    )
    op.create_index('idx_daily_usage_user_date', 'daily_usage', ['user_id', 'usage_date'], unique=False)

    # Create location_visits table
    op.create_table('location_visits',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('longitude', sa.DECIMAL(precision=10, scale=8), nullable=False),
        sa.Column('latitude', sa.DECIMAL(precision=11, scale=8), nullable=False),
        sa.Column('address', sa.Text(), nullable=True),
        sa.Column('names', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('visit_time', sa.DateTime(), nullable=False),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_location_visits_user_time', 'location_visits', ['user_id', 'visit_time'], unique=False)
    op.create_index('idx_location_visits_coordinates', 'location_visits', ['longitude', 'latitude'], unique=False)
    # Create GIN index for array field
    op.execute('CREATE INDEX idx_location_visits_names ON location_visits USING GIN(names)')

    # Create text_notes table
    op.create_table('text_notes',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('text_content', sa.Text(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('longitude', sa.DECIMAL(precision=10, scale=8), nullable=True),
        sa.Column('latitude', sa.DECIMAL(precision=11, scale=8), nullable=True),
        sa.Column('address', sa.Text(), nullable=True),
        sa.Column('names', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_text_notes_user_time', 'text_notes', ['user_id', 'timestamp'], unique=False)
    # Create GIN index for array field
    op.execute('CREATE INDEX idx_text_notes_names ON text_notes USING GIN(names)')

    # Create media_files table
    op.create_table('media_files',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('file_type', sa.String(length=50), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('longitude', sa.DECIMAL(precision=10, scale=8), nullable=True),
        sa.Column('latitude', sa.DECIMAL(precision=11, scale=8), nullable=True),
        sa.Column('address', sa.Text(), nullable=True),
        sa.Column('names', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    # Create GIN index for array field
    op.execute('CREATE INDEX idx_media_files_names ON media_files USING GIN(names)')

    # Create query_sessions table
    op.create_table('query_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_context', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('last_query', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_query_sessions_user_expires', 'query_sessions', ['user_id', 'expires_at'], unique=False)

    # Insert sample data for testing
    op.execute("""
        INSERT INTO users (id, email, oauth_provider, oauth_subject, display_name, subscription_tier, created_at, updated_at)
        VALUES 
        ('550e8400-e29b-41d4-a716-446655440000', 'test@example.com', 'google', 'google_123456', 'Test User', 'free', NOW(), NOW()),
        ('550e8400-e29b-41d4-a716-446655440001', 'premium@example.com', 'apple', 'apple_789012', 'Premium User', 'premium', NOW(), NOW()),
        ('550e8400-e29b-41d4-a716-446655440002', 'pro@example.com', 'google', 'google_345678', 'Pro User', 'pro', NOW(), NOW())
    """)

    # Insert sample location visits
    op.execute("""
        INSERT INTO location_visits (id, user_id, longitude, latitude, address, names, visit_time, duration, description, created_at, updated_at)
        VALUES 
        ('660e8400-e29b-41d4-a716-446655440000', '550e8400-e29b-41d4-a716-446655440000', -122.4194, 37.7749, '123 Market St, San Francisco, CA', ARRAY['Office', 'Work'], '2024-01-15 09:00:00', 480, 'Morning work session', NOW(), NOW()),
        ('660e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440000', -122.4089, 37.7849, '456 Union Square, San Francisco, CA', ARRAY['Shopping', 'Mall'], '2024-01-15 18:30:00', 120, 'Evening shopping', NOW(), NOW())
    """)

    # Insert sample text notes
    op.execute("""
        INSERT INTO text_notes (id, user_id, text_content, timestamp, longitude, latitude, address, names, created_at)
        VALUES 
        ('770e8400-e29b-41d4-a716-446655440000', '550e8400-e29b-41d4-a716-446655440000', 'Had a great meeting with the team today. Discussed the new project roadmap.', '2024-01-15 14:30:00', -122.4194, 37.7749, '123 Market St, San Francisco, CA', ARRAY['Office', 'Work'], NOW()),
        ('770e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', 'Beautiful sunset at the beach. Perfect end to the day.', '2024-01-15 19:45:00', -122.4830, 37.8199, 'Ocean Beach, San Francisco, CA', ARRAY['Beach', 'Sunset'], NOW())
    """)

    # Insert sample daily usage
    op.execute("""
        INSERT INTO daily_usage (id, user_id, usage_date, text_notes_count, media_files_count, queries_count, created_at, updated_at)
        VALUES 
        ('880e8400-e29b-41d4-a716-446655440000', '550e8400-e29b-41d4-a716-446655440000', '2024-01-15', 2, 1, 5, NOW(), NOW()),
        ('880e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', '2024-01-15', 1, 0, 3, NOW(), NOW())
    """)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop tables in reverse order due to foreign key constraints
    op.drop_table('query_sessions')
    op.drop_table('media_files')
    op.drop_table('text_notes')
    op.drop_table('location_visits')
    op.drop_table('daily_usage')
    op.drop_table('users')
