from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = 'aff5d4662e81'
down_revision = '7eb0dc8b6600'
branch_labels = None
depends_on = None

def upgrade():
    bind = op.get_bind()
    insp = inspect(bind)

    # Look at existing columns on claim_request
    cols = [c['name'] for c in insp.get_columns('claim_request')]

    # Only add these if they’re not already there:
    if 'is_successful' not in cols:
        op.add_column('claim_request',
            sa.Column('is_successful', sa.Boolean(), nullable=True)
        )
    if 'in_person_required' not in cols:
        op.add_column('claim_request',
            sa.Column('in_person_required', sa.Boolean(), nullable=True)
        )
    if 'in_person_deadline' not in cols:
        op.add_column('claim_request',
            sa.Column('in_person_deadline', sa.DateTime(), nullable=True)
        )
    # …and so on for your in-person fields…


def downgrade():
    pass
