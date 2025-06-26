from datetime import datetime
from flask_login import UserMixin
from sqlalchemy import func, or_
from extensions import db # Import db from extensions.py

# --- Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String, unique=True, nullable=False)
    password_hash = db.Column(db.String, nullable=False)
    roles = db.Column(db.String, nullable=False, default='')

    @property
    def is_admin(self):
        return 'admin' in self.roles.split(',')
    @property
    def success_rate(self):
        # Count all of this user’s finalized claims
        total = ClaimRequest.query.\
            filter(ClaimRequest.user_id == self.id,
                   ClaimRequest.is_successful != None).count()
        if total == 0:
            # No history yet → treat as “perfect” so they aren’t penalized
            return 1.0
        # Count only their successful ones
        wins = ClaimRequest.query.\
            filter_by(user_id=self.id, is_successful=True).count()
        # Return a float between 0 and 1
        return wins / total


class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String, nullable=False)
    description = db.Column(db.String, nullable=False)
    location = db.Column(db.String, nullable=False)
    date_found = db.Column(db.String, nullable=False)
    category = db.Column(db.String, nullable=False)
    contact = db.Column(db.String, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    claimed = db.Column(db.Boolean, default=False)
    claimed_by = db.Column(db.String, nullable=True)
    received = db.Column(db.Boolean, default=False)
    img_emb = db.Column(db.PickleType)
    priority_score = db.Column(db.Float, default=0.0)
    qr_code = db.Column(db.String, nullable=True)
    match_candidates = db.Column(db.JSON, nullable=True)  # top 3 lost matches
    is_container = db.Column(db.Boolean, default=False)
    container_contents = db.Column(db.Text, nullable=True)


class LostItem(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_email  = db.Column(db.String, nullable=False)               # pulled from current_user
    description = db.Column(db.String, nullable=False)
    date_lost   = db.Column(db.Date,   nullable=False)
    location    = db.Column(db.String, nullable=False)
    contact     = db.Column(db.String, nullable=False)               # phone
    has_images  = db.Column(db.Boolean, default=False)               # “Do you have images?”
    img_emb     = db.Column(db.PickleType, nullable=True)            # optional embedding
    timestamp   = db.Column(db.DateTime, server_default=db.func.now())
    received = db.Column(db.Boolean, default=False)
    category = db.Column(db.String, nullable=False)


class Photo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('report.id'), nullable=False)
    filename = db.Column(db.String, nullable=False)

    # Relationship back to Report
    report = db.relationship('Report', backref=db.backref('photos', lazy=True))


class ClaimRequest(db.Model):
    __table_args__ = (
        db.Index('ix_claim_req_report_created', 'report_id', 'created_at'),
        db.Index('ix_claim_req_user_email', 'user_email'),
    )
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref='claims')

    report_id = db.Column(db.Integer, db.ForeignKey('report.id'), nullable=False)
    status = db.Column(db.String, nullable=False, default='pending')
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    report = db.relationship('Report', backref=db.backref('claim_requests', lazy=True))
    description = db.Column(db.String)
    location1 = db.Column(db.String)
    location2 = db.Column(db.String)
    location3 = db.Column(db.String)
    quality_score = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    delay_seconds = db.Column(db.Float, nullable=True)
    proof_type = db.Column(db.String(20), nullable=True)
    ocr_text = db.Column(db.Text, nullable=True)
    image_features = db.Column(db.PickleType, nullable=True)
    proof_score = db.Column(db.Float, nullable=True)
    proof_filename = db.Column(db.String, nullable=True)
    proof_deadline = db.Column(db.DateTime, nullable=True)
    decision     = db.Column(db.String, nullable=True)  # 'accept' or 'decline'
    finalized    = db.Column(db.Boolean, default=False)  # True after 10 minutes
    pending_until = db.Column(db.DateTime, nullable=True)  # Deadline to undo
    is_successful = db.Column(db.Boolean, nullable=True)
    # In-person verification fields
    in_person_required = db.Column(db.Boolean, default=False)
    in_person_deadline = db.Column(db.DateTime, nullable=True)
    in_person_notes = db.Column(db.Text, nullable=True)
    in_person_requested_by = db.Column(db.String, nullable=True) # Admin email
    in_person_requested_at = db.Column(db.DateTime, nullable=True)
    in_person_verified = db.Column(db.Boolean, default=False)

class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('report.id', name='fk_complaint_report'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', name='fk_complaint_user'), nullable=True)
    report = db.relationship('Report', backref=db.backref('complaints', lazy=True))
    details = db.Column(db.Text, nullable=False)
    proof_filename = db.Column(db.String(255), nullable=True)
    quality_score = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='complaints', lazy=True)
    proof_type = db.Column(db.String(20), nullable=True)
    ocr_text = db.Column(db.Text, nullable=True)
    image_features = db.Column(db.PickleType, nullable=True)
    proof_score = db.Column(db.Float, nullable=True)
    counter_filename = db.Column(db.String(255), nullable=True)
    counter_score = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), nullable=False, default='pending')


class OwnershipEvent(db.Model):
    __tablename__ = 'ownership_event'
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('report.id'), nullable=False)
    from_email = db.Column(db.String, nullable=False)
    to_email = db.Column(db.String, nullable=False)
    reason = db.Column(db.String, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    report = db.relationship('Report', backref=db.backref('ownership_events', lazy=True))

class ThankYouMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    finder_email = db.Column(db.String(120), nullable=False)
    report_id = db.Column(db.Integer, db.ForeignKey('report.id'))
    claimant_email = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Announcement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class NotificationRead(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String, nullable=False)
    announcement_id = db.Column(db.Integer, db.ForeignKey('announcement.id'), nullable=True)
    thank_you_id = db.Column(db.Integer, db.ForeignKey('thank_you_message.id'), nullable=True)
    read_at = db.Column(db.DateTime, default=datetime.utcnow)