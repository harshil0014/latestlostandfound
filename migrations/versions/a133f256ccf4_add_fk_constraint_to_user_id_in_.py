"""Add FK constraint to user_id in ClaimRequest

Revision ID: a133f256ccf4
Revises: aff5d4662e81
Create Date: 2025-06-23 04:26:34.479577

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a133f256ccf4'
down_revision = 'aff5d4662e81'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('claim_request', schema=None) as batch_op:
        batch_op.alter_column('user_id',
               existing_type=sa.INTEGER(),
               nullable=False)
        batch_op.create_foreign_key(
            'fk_claim_request_user_id',  # Give it a name!
            'user',                      # Referenced table
            ['user_id'],                 # Local column
            ['id']                       # Remote column
        )



def downgrade():
    with op.batch_alter_table('claim_request', schema=None) as batch_op:
        batch_op.drop_constraint('fk_claim_request_user_id', type_='foreignkey')
        batch_op.alter_column('user_id',
               existing_type=sa.INTEGER(),
               nullable=True)

