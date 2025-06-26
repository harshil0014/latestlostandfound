"""Mark container field migration as applied"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'fe7509b83f3c'
down_revision = 'aff5d4662e81'  # or your correct parent
branch_labels = None
depends_on = None

def upgrade():
    pass

def downgrade():
    pass
