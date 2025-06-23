"""add complaint model

Revision ID: 466da0e2ed8a
Revises: 
Create Date: 2025-06-15 13:05:58.777382
"""

from alembic import op
import sqlalchemy as sa

revision = '466da0e2ed8a'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('complaint',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column('proof_filename', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('details', sa.Text(), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='pending'), # Example: default status 'pending'
        sa.Column('proof_type', sa.String(100), nullable=True),
        sa.Column('proof_score', sa.Float(), nullable=True),
        sa.Column('ocr_text', sa.Text(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('image_features', sa.PickleType(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], name='fk_complaint_user')
    )
    # Note: Ensure the 'user' table (referenced by the foreign key)
    # exists or is created by a preceding migration.


def downgrade():
    # The downgrade operation for create_table is drop_table.
    # If a schema was specified in op.create_table, it should also be specified here.
    op.drop_table('complaint')

