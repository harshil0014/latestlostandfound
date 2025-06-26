import os
import sys
import re
from dotenv import load_dotenv
from datetime import date, timedelta, datetime
from functools import wraps
import numpy as np
from threading import Thread
import pickle
import faiss
import tempfile
from wtforms import MultipleFileField
from dateutil.relativedelta import relativedelta  
from logging import INFO # Import INFO for logger level
from PIL import UnidentifiedImageError
from uuid import uuid4


from flask import (
    Flask, request, redirect, url_for, flash, render_template,
    session, abort, jsonify
)
from flask import current_app
from sqlalchemy.orm import joinedload
from flask_migrate import Migrate
from flask_wtf import FlaskForm, CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

csrf = CSRFProtect()

from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user, UserMixin
)
from wtforms import (  # type: ignore
    BooleanField, HiddenField, StringField, SelectField,
    DateField, FileField, SubmitField, TelField, TextAreaField
)
from wtforms.validators import DataRequired, Length, Regexp
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import func, or_
import bleach
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from uuid import uuid4
from vector_store import VectorStore # Keep this import for VectorStore class
import qrcode
import click # Added for CLI commands

from ai_utils import get_image_vec, get_text_vec, cosine_sim, ocr_image_to_text
from extensions import db

# immediately after you import db:
from models import User, Report, LostItem, Photo, ClaimRequest, Complaint, \
                   OwnershipEvent, ThankYouMessage, Announcement, NotificationRead

# Load environment variables
load_dotenv()

limiter = Limiter(
    key_func=lambda: current_user.email or get_remote_address(),
    default_limits=[],
    storage_uri="memory://"
)

# --- Module-level LoginManager instance ---
login_manager = LoginManager()
login_manager.login_view = 'login'

# Ensure SendGrid key is available
SENDGRID_KEY = os.getenv('SENDGRID_API_KEY')

# AI thresholds
AI_MATCH_THRESHOLD = 0.23
DUPLICATE_THRESHOLD = 0.90
# Urgent keywords for priority
URGENT_KEYWORDS = [
    "passport", "visa", "epipen", "epi-pen", "insulin", "inhaler",
    "medical", "prescription", "hearing aid", "oxygen", "oxygen tank",
]

# Items so common they get down-weighted
COMMON_ITEMS = {
    'pen', 'keys', 'wallet', 'phone', 'notebook', 'book', 'card', 'bag'
}

# Helper functions
def finalize_expired_claims():
    now = datetime.utcnow()
    pending_claims = ClaimRequest.query.filter(
        ClaimRequest.finalized == False,
        ClaimRequest.pending_until <= now
    ).all()

    for cr in pending_claims:
        decision = cr.decision  # 'accept' or 'decline'
        cr.status = decision
        cr.finalized = True

        if decision == 'accept':
            rpt = cr.report
            rpt.claimed = True
            rpt.claimed_by = cr.user_email

            # QR code
            qr_rel_path = generate_qr(f"Found item #{rpt.id} claimed by {cr.user_email}")
            rpt.qr_code = qr_rel_path

            # Thank you
            if rpt.user_email and rpt.user_email != cr.user_email:
                thank_you = ThankYouMessage(
                    finder_email=rpt.user_email,
                    report_id=rpt.id,
                    claimant_email=cr.user_email
                )
                db.session.add(thank_you)

            # Decline others
            others = ClaimRequest.query.filter(
                ClaimRequest.report_id == cr.report_id,
                ClaimRequest.id != cr.id
            )
            for o in others:
                o.status = 'declined'
                o.finalized = True

    db.session.commit()

def compute_quality_score(text: str) -> float:
    text = text.lower().strip()
    words = text.split()
    length = min(len(words) / 12, 0.4)
    has_col = 0.3 if re.search(r'\b(black|white|red|blue|green|brown|grey)\b', text) else 0
    has_brand = 0.3 if re.search(r'\b(casio|hp|nike|titan|sony|apple|samsung|lenovo)\b', text) else 0
    has_num = 0.2 if re.search(r'\b\d{3,}\b', text) else 0
    has_kw = 0.2 if re.search(r'\b(wallet|phone|book|id|card|pen|earphones|calculator|bag)\b', text) else 0
    has_loc = 0.2 if re.search(r'\b(library|canteen|lab|class|hall|ground)\b', text) else 0
    return min(length + has_col + has_brand + has_num + has_kw + has_loc, 1.0)


def compute_content_match(admin_contents: str, claim_contents: str) -> float:
    """
    Returns a float 0–1 for how well the claimant’s list matches
    the admin’s container_contents list, with uncommon items boosted.
    Both inputs are comma-separated strings.
    """
    # Normalize & split
    admin_items = [i.strip().lower() for i in admin_contents.split(',') if i.strip()]
    claim_items = {i.strip().lower() for i in claim_contents.split(',') if i.strip()}

    # Compute weighted totals
    total_weight = 0.0
    matched_weight = 0.0

    for item in admin_items:
        weight = 1.5 if item not in COMMON_ITEMS else 0.5
        total_weight += weight
        if item in claim_items:
            matched_weight += weight

    return (matched_weight / total_weight) if total_weight > 0 else 0.0



def compute_priority(description: str) -> float:
    desc_lc = description.lower()
    for w in URGENT_KEYWORDS:
        if w in desc_lc:
            return 1.0
    return 0.0


def predict_category(description):
    desc = description.lower()
    if any(word in desc for word in ["book", "novel", "pages", "notes"]):
        return "books"
    elif any(word in desc for word in ["wallet", "purse", "card", "cash", "bag","backpack","rucksack","tote"]):
        return "accessories"
    elif any(word in desc for word in ["id", "pan", "license", "aadhar"]):
        return "identity"
    elif any(word in desc for word in ["pen", "pencil", "sharpener"]):
        return "stationary"
    else:
        return "others"


def generate_qr(data: str) -> str:
    img = qrcode.make(data)
    fname = f"{uuid4().hex}.png"
    rel_path = f"qr/{fname}"
    abs_folder = current_app.config['QR_FOLDER']
    os.makedirs(abs_folder, exist_ok=True)
    abs_path = os.path.join(abs_folder, fname)
    img.save(abs_path)
    current_app.logger.info(f"QR generated → {rel_path}")
    return rel_path


def async_index_report(photo_paths, rpt_id):
    """Compute image vectors and .add() in one go, off the request thread."""
    with app.app_context():
        vecs = []
        for p in photo_paths:
            try:
                vecs.append(get_image_vec(p))
            except Exception as e:
                current_app.logger.error(f"Error processing image {p}: {e}")
                pass # Skip problematic images
        
        img_vec = None
        if vecs:
            mat = np.vstack(vecs).astype('float32')
            img_vec = np.mean(mat, axis=0)

        # Update the Report's img_emb in the database
        rpt = Report.query.get(rpt_id)
        if rpt:
            rpt.img_emb = img_vec
            db.session.commit()
            current_app.logger.info(f"Report {rpt_id} img_emb updated in background.")

            # Now add to FAISS
            if img_vec is not None:
                is_new_item = current_app.vs.add(img_vec, rpt_id, DUPLICATE_THRESHOLD)
                if not is_new_item:
                    current_app.logger.warning(f"Report {rpt_id} identified as duplicate during background indexing.")
        else:
            current_app.logger.error(f"Report {rpt_id} not found for background indexing.")


# Flask application factory
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///lost_and_found.db')
    app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
    app.config['QR_FOLDER'] = os.path.join('static', 'qr')
    # AI/FAISS configuration
    app.config['EMB_DIM'] = 512 # Dimension of your embeddings (e.g., from clip-ViT-B-32)
    app.config['VS_INDEX_PATH'] = os.path.join(app.instance_path, 'faiss_index.bin') # Persistent storage for FAISS index
    # Flask-Limiter Storage Configuration: Use Redis for persistent rate limits
    app.config['RATELIMIT_STORAGE_URL'] = 'memory://' # Use in-memory storage for development
    app.logger.setLevel(INFO) # Set logger level for app

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['QR_FOLDER'], exist_ok=True)

    db.init_app(app)
    migrate = Migrate(app, db)
    csrf.init_app(app)
        # Initialize login manager with the app
    login_manager.init_app(app)

    limiter.init_app(app) # Hook our existing module-level limiter into this app

    # ── Build FAISS store inside an app context so `current_app` works ──
    with app.app_context():
        # Initialize your VectorStore and attach it to the app.
        # This ensures app.vs is always available when the app is created.
        # The VectorStore constructor takes index_path, and accesses EMB_DIM from app.config internally.
        emb_dim = app.config['EMB_DIM']
        faiss_path = app.config['VS_INDEX_PATH']
        app.vs = VectorStore(index_path=faiss_path, emb_dim=emb_dim)

    return app

# Create and configure app
app = create_app()

# User loader callback
@login_manager.user_loader
def load_user(user_id):
    # we’ve already imported User inside create_app, so it’s available here
    return User.query.get(int(user_id))





# ─── Forms ─────────────────────────────────────────────────────────────────
class ReportForm(FlaskForm):
    description = StringField('Description', validators=[
        DataRequired(), Length(max=100),
        Regexp(r'^[A-Za-z0-9\s\.!\-]+$',
               message="Only letters, numbers, spaces, and .!- allowed.")
    ])
    location = StringField('Location', validators=[
        DataRequired(), Length(max=50),
        Regexp(r'^[A-Za-z0-9\s,\-]+$',
               message="Only letters, numbers, spaces, commas, dashes allowed.")
    ])
    date_found = DateField('Date Found', validators=[DataRequired()], format='%Y-%m-%d')
    category = SelectField('Category', validators=[DataRequired()], choices=[
        ('accessories','Accessories'),
        ('books','Books'),
        ('stationary','Stationary'),
        ('others','Others')
    ])
    contact = TelField('Contact', validators=[
        DataRequired(), Length(max=20),
        Regexp(r'^\d{10}$', message="Only digits, +, -, spaces allowed.")
    ])
    photos = MultipleFileField(
        'Photos',
        validators=[DataRequired()],
        render_kw={'multiple': True}
    )
    is_container = BooleanField('Is this a container (e.g., bag, wallet)?')
    container_contents = TextAreaField(
        'If yes, list contents (comma-separated)',
        validators=[Length(max=500)],
        render_kw={'placeholder': 'e.g., ID card, keys, pen, 200 rupees'}
    )

class LostReportForm(FlaskForm):
    description = StringField('What did you lose?', validators=[
        DataRequired(), Length(max=100),
        Regexp(r'^[A-Za-z0-9\s\.\-]+$', message="Only letters, numbers, spaces, and .- allowed.")
    ])
    date_lost = DateField('When did you lose it?', validators=[DataRequired()], format='%Y-%m-%d')
    location1 = StringField('Possible Location 1', validators=[DataRequired(), Length(max=50)])
    location2 = StringField('Possible Location 2', validators=[Length(max=50)])
    location3 = StringField('Possible Location 3', validators=[Length(max=50)])
    category = SelectField('Item Category', validators=[DataRequired()], choices=[
        ('accessories', 'Accessories'),
        ('books', 'Books'),
        ('stationary', 'Stationary'),
        ('others', 'Others')
    ])
    contact = TelField('Your Contact Number', validators=[
        DataRequired(), Length(max=20),
        Regexp(r'^\d{10}$', message="Only 10-digit numbers allowed.")
    ])
    submit = SubmitField('Submit Lost Report')



class ComplaintForm(FlaskForm):
    report_id = HiddenField()
    details = TextAreaField(
        "Why this item belongs to you",
        validators=[DataRequired(), Length(min=30)]
    )
    proof = FileField("Proof (image / PDF)")
    submit = SubmitField("Submit Complaint")


# ─── Context Processors & Decorators ─────────────────────────────────────

def admin_only(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return wrapper



# ─── Routes & CLI Commands ─────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        pw = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, pw):
            login_user(user)
            return redirect(request.args.get('next') or url_for('show_home'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')

import click
from flask.cli import with_appcontext

@app.cli.command("check_abuse")
@with_appcontext
def check_abuse():
    """Print users with ≥5 low-quality claims in last 24 h."""
    from datetime import datetime, timedelta
    day_ago = datetime.utcnow() - timedelta(hours=24)

    rows = (db.session.query(
                ClaimRequest.user_email.label("email"),
                db.func.count().label("total"),
                db.func.avg(ClaimRequest.quality_score).label("avg_q")
            )
            .filter(ClaimRequest.created_at >= day_ago)
            .group_by(ClaimRequest.user_email)
            .having(db.func.count() >= 3)
            .having(db.func.avg(ClaimRequest.quality_score) <= 0.4)
            .all())

    if not rows:
        click.echo("✅ No abusive users in last 24 h.")
    else:
        click.echo("⚠️  Suspicious users:")
        for email, total, avg in rows:
            click.echo(f"  {email} – {total} claims, avg score {avg:.2f}")


@app.route('/allow-reports')
@admin_only
def allow_reports():

    pending_reports = Report.query.filter_by(received=False) \
        .order_by(Report.priority_score.desc(), Report.timestamp.desc()) \
        .all()

    # ── Compute waiting time for each report ──
    now = datetime.utcnow()
    for rpt in pending_reports:
        delta = now - rpt.timestamp
        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60

        if days:
            rpt.wait_time = f"{days}d {hours}h"
        elif hours:
            rpt.wait_time = f"{hours}h"
        else:
            rpt.wait_time = f"{minutes}m"

        rpt.is_container = getattr(rpt, "is_container", False)
        rpt.container_contents = getattr(rpt, "container_contents", "")


    return render_template('allow_reports.html',reports=pending_reports)


@app.route('/allow-report/<int:report_id>/accept', methods=['POST'])
@admin_only
def accept_report(report_id):
    rpt = Report.query.get_or_404(report_id)
    rpt.received = True
    db.session.commit()
    return ('', 204)

@app.route('/allow-report/<int:report_id>/delete', methods=['POST'])
@admin_only
def delete_unapproved_report(report_id):
    rpt = Report.query.get_or_404(report_id)
    ClaimRequest.query.filter_by(report_id=report_id).delete()
    Complaint.query.filter_by(report_id=report_id).delete()
    db.session.delete(rpt)
    db.session.commit()
    return ('', 204)

# --------------------------------------------------
# Route: file an ownership complaint for a report
# URL:   /complain/<report_id>
# --------------------------------------------------
@app.route("/complain/<int:report_id>", methods=["GET","POST"])
@login_required
def complain(report_id):
    report = Report.query.get_or_404(report_id)
    form = ComplaintForm()
    form.report_id.data = report_id

    proof_file = form.proof.data
    proof_filename = None
    proof_type = None
    path = None

    if form.validate_on_submit():
        # 🚫 File size limit: max 5 MB per photo
        MAX_SIZE = 5 * 1024 * 1024  # bytes

        if proof_file:
            # measure size
            photo.stream.seek(0, os.SEEK_END)
            size = photo.stream.tell()
            photo.stream.seek(0)
            if size > MAX_SIZE:
                flash('Each photo must be under 5 MB.', 'warning')
                return redirect(url_for('report_found'))
            
        # 🚫 Only allow these extensions
            ALLOWED = {'jpg', 'jpeg', 'png', 'pdf'}
            if proof_file:
                ext = proof_file.filename.rsplit('.', 1)[-1].lower()
                if ext not in ALLOWED:
                    flash('Only JPG, JPEG, PNG, or PDF files allowed.', 'warning')
                    return redirect(url_for('report_found'))

        
        # 🚫 Limit to 5 photos max
        if len(form.photos.data) > 5:
            flash('You can upload up to 5 photos only.', 'warning')
            return redirect(url_for('report_found'))

        # 1. save the upload
        if proof_file:
            proof_filename = secure_filename(proof_file.filename)
            proof_path = os.path.join(
                                app.config["UPLOAD_FOLDER"],
                                "complaints",
                                proof_filename)
            os.makedirs(os.path.dirname(proof_path), exist_ok=True)
            proof_file.save(proof_path)
            path = proof_path

            proof_type = ('image' if proof_filename.lower().endswith(('.png','.jpg','.jpeg'))else 'receipt')

        # 2. build & flush the Complaint
        complaint = Complaint(
            report_id = report.id,
            user_id = current_user.id,
            details = form.details.data,
            proof_filename= proof_filename,
            quality_score = min(len(form.details.data.strip())/200.0, 1.0)
        )
        db.session.add(complaint)
        db.session.flush()

        # 3. AI‐scoring (only if there was an upload)
        if proof_filename:
            if proof_type == 'image':
                try:
                    vec = get_image_vec(path)
                    rpt_vec = complaint.report.img_emb if complaint.report.img_emb is not None else []
                    complaint.proof_score = cosine_sim(vec, rpt_vec)
                except (FileNotFoundError, UnidentifiedImageError):
                    flash("Invalid image file. Please upload a valid image.", "danger")
                    return render_template("complaint_form.html", form=form)
            elif proof_type == 'receipt':
                complaint.ocr_text = ocr_image_to_text(path)
                complaint.proof_score = compute_content_match(complaint.ocr_text, report.description)

            # 4. finalize & commit
            complaint.status = 'proof submitted'
            # complaint.proof_type = proof_type # already set above
            db.session.commit()

            flash("Proof uploaded! Admin will review soon.", "success")
            return redirect(url_for('my_claims'))

    # GET (or invalid POST) → show form again
    return render_template("complaint_form.html", form=form)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/help')
@login_required
def help():
    return render_template('help.html')


@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')


@app.route('/report-found', methods=['GET','POST'])
@login_required
@limiter.limit("3 per day")   # temporarily disable rate-limit
def report_found():
    #  1️⃣ rate-limit check
    # today = date.today().isoformat()
    # existing = Report.query.filter_by(
    #     email=current_user.email,
    #     date_found=today
    # ).count()
    # if existing >= 3:
    #     flash("You've already submitted 3 found-item reports today. Please try again tomorrow.", "warning")
    #     return redirect(url_for('show_home'))

    form = ReportForm()
    if form.validate_on_submit():
        predicted = predict_category(form.description.data)
        if form.category.data != predicted:
            flash(f"This item was stored under '{predicted.title()}' instead of '{form.category.data.title()}' because it matched better.")
            form.category.data = predicted

        # 2. Create the Report (without a filename; embedding later)
        rpt = Report(
            email=current_user.email,
            description=bleach.clean(form.description.data), # ← sanitize
            location=form.location.data,
            date_found=form.date_found.data.strftime('%Y-%m-%d'),
            category=form.category.data,
            contact=form.contact.data,
            received=False,
            priority_score=compute_priority(form.description.data),
            is_container=form.is_container.data,
            container_contents=bleach.clean(form.container_contents.data) if form.is_container.data else None,
        )
        db.session.add(rpt)
        db.session.flush()     # ← now rpt.id exists

        # 🔍 Step: Match against existing lost items
        found_desc_vec = get_text_vec(rpt.description)
        found_loc_vec = get_text_vec(rpt.location)
        
        lost_items = LostItem.query.filter_by(received=True).all()
        # 1) Build a list of (lost_obj, score) tuples:
        pairs = []
        
        for lost in lost_items:
            desc_sim = cosine_sim(found_desc_vec, get_text_vec(lost.description))
            loc_sim  = cosine_sim(found_loc_vec, get_text_vec(lost.location))
            # 70% from text, 30% from location
            avg_score = round(desc_sim * 0.7 + loc_sim * 0.3, 3)
            if avg_score > 0.35: # 💡 tune this threshold
                pairs.append((lost, avg_score))

        # 2) Take top 3:
        top = sorted(pairs, key=lambda x: x[1], reverse=True)[:3]

        # 3) Store them in the Report:
        rpt.match_candidates = [
            {
            "lost_id": lost.id,
            "description": lost.description,
            "location": lost.location,
            "match_score": score
            }
            for lost, score in top
        ]
        # 3) Store top-3 matches (if any) so we can notify later 
        if top:
            rpt.match_candidates = [
                {
                   "lost_id": lost.id,
                   "description": lost.description,
                   "location": lost.location,
                   "match_score": score
                }
                for lost, score in top  ]
        else:
            rpt.match_candidates = None



        report_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(rpt.id))
        os.makedirs(report_folder, exist_ok=True) # ← move folder creation here

        # 1. Save uploaded photos into that folder
        filenames = []
        for photo in form.photos.data:
            if not photo.mimetype.startswith('image/'):
                continue
            if photo and photo.filename: # ← check
                fname = f"{uuid4().hex}_{secure_filename(photo.filename)}"
                path = os.path.join(report_folder, fname)           # ← save into report_folder
                photo.save(path)
                filenames.append(fname)

        # 🔍 Debug: confirm
        print(f"Photo files saved in {report_folder}: {filenames}")



        # 3. Create Photo rows
        for fname in filenames:
            rel = f"{rpt.id}/{fname}"
            db.session.add(Photo(report_id=rpt.id, filename=rel))


        # 4. Commit both report + photos
        db.session.commit()



        # ── Compute embedding from all uploaded photos ──
        img_vecs = []
        for fname in filenames:
            path = os.path.join(report_folder, fname)

            try:
                vec = get_image_vec(path)
                img_vecs.append(vec)
            except Exception:
                pass  # skip any photo that fails

        if img_vecs:
            # average all vectors into one
            stacked = np.vstack(img_vecs).astype('float32')
            img_vec = np.mean(stacked, axis=0)
        else: # 💡 what if no images?
            img_vec = None
        
        # — Store into the Report for later AI checks and FAISS index —
        rpt.img_emb = img_vec
        db.session.commit()
        # — Add this report’s embedding into the FAISS index —
        if img_vec is not None:
            try:
                import threading
                photo_paths = [os.path.join(app.config['UPLOAD_FOLDER'], str(rpt.id), f) for f in filenames]
                thread = threading.Thread(target=async_index_report, args=(photo_paths, rpt.id))
                thread.daemon = True
                thread.start()
            except Exception:
                current_app.logger.exception("🔥 Failed to start background index thread")
        desc = bleach.clean(form.description.data) # ← sanitize
        # ---- DESCRIPTION-vs-PHOTO GUARD ----
        if img_vec is not None:
            text_vec = get_text_vec(desc)
            similarity = cosine_sim(img_vec, text_vec)
            print(f"Mismatch-check sim={similarity:.3f}")
            if similarity < AI_MATCH_THRESHOLD:
                flash("The photo doesn’t seem to match the description. Please revise.", "warning")
                return redirect(url_for('report_found'))
        # ---- END GUARD ----        

        flash('Report submitted!', 'success')
        return redirect(url_for('category_items', cat=form.category.data))
    return render_template('report_found.html', form=form)

@app.route('/report-lost', methods=['GET', 'POST'])
@login_required
@limiter.limit("3 per day")
def report_lost():
    # rate-limit: max 3 lost-item reports per day
    today = date.today()
    existing = LostItem.query.filter(
        LostItem.user_email==current_user.email,
        LostItem.date_lost==today
    ).count()
    if existing >= 3:
        flash("You've already submitted 3 lost-item reports today. Please try again tomorrow.", "warning")
        return redirect(url_for('show_home'))

    form = LostReportForm()

    if form.validate_on_submit():
        item = LostItem(
            user_email=current_user.email,
            description=form.description.data,
            date_lost=form.date_lost.data,
            location=form.location1.data,
            contact=form.contact.data,
            category=form.category.data, 
            has_images=False,
            img_emb=None,
            received=False

        )
        db.session.add(item)
        db.session.commit()
        flash('Lost item report submitted!', 'success')
        return redirect(url_for('lost_items'))

    return render_template('report_lost.html', form=form)




@app.route('/found-items')
@login_required
def found_items():
    accepted = ClaimRequest.query.filter(
        or_(
          ClaimRequest.status == 'accepted',
          ClaimRequest.status == 'requires proof'
        )
    ).all()
    return render_template('itemsfound.html', accepted=accepted)


@app.route('/category/<cat>')
@login_required
def category_items(cat):
    filter_date = request.args.get('filter_date')
    q = db.session.query(Report).filter(
        func.lower(Report.category) == cat.lower(),
        Report.received == True
    ).order_by(Report.timestamp.desc())

    if filter_date:
        q = q.filter_by(date_found=filter_date)
    items = q.order_by(Report.timestamp.desc()).all()

    items_for_js = [
        {'id': i.id, 'description': i.description, 'location': i.location}
        for i in items
    ]

    all_dates = [r.date_found for r in Report.query.with_entities(Report.date_found)]
    max_date = date.today().isoformat()
    min_date = (date.today() - timedelta(days=6)).isoformat()

    user_claims = {
        cr.report_id for cr in ClaimRequest.query.filter_by(user_email=current_user.email)
    }

    return render_template(
        'categoryitems.html',
        items=items,
        items_for_js=items_for_js,
        category=cat,
        email=current_user.email,
        roles=current_user.roles.split(","),
        min_date=min_date,
        max_date=max_date,
        filter_date=filter_date,
        user_claims=user_claims,
        success_rate=current_user.success_rate
    )

# AJAX‐style Claim → JSON
@app.route('/claim/<int:report_id>', methods=['POST'])
@csrf.exempt
@login_required
def claim_report(report_id):
     # If user’s success rate is under 50%, require a video
    if current_user.success_rate < 0.5:
        if 'video_file' not in request.files or request.files['video_file'].filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Because your past claims are under 50% accurate, please upload a short video.'
            }), 400

    exists = ClaimRequest.query.filter_by(
        user_email=current_user.email,
        report_id=report_id
    ).first()


    if not exists:
            reason = request.form.get('description', '')
            q_score = compute_quality_score(reason)

            loc1 = request.form.get('location1', '')
            loc2 = request.form.get('location2', '')

            report = Report.query.get(report_id)
            # 1a. Handle uploaded proof file
            proof_file = request.files.get('proof_file')
            proof_filename = None
            if proof_file:
                proof_filename = secure_filename(proof_file.filename)
                proof_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'claims')
                os.makedirs(proof_folder, exist_ok=True)
                proof_path = os.path.join(proof_folder, proof_filename)
                proof_file.save(proof_path)

            delay = (datetime.utcnow() - report.timestamp).total_seconds()

            
            cr = ClaimRequest(
                    report_id = report_id,
                    user_email = current_user.email,
                    description = reason,
                    location1     = loc1,
                    location2     = loc2,
                    quality_score = q_score,    # <-- NEW
                    created_at = datetime.utcnow(),
                    delay_seconds = delay,
                    proof_filename=proof_filename
                )

            db.session.add(cr)
            db.session.commit()
            return jsonify(message='Your claim request has been sent!')
    else:
        return jsonify(message='Already requested.')

# AJAX‐style Delete → 204 No Content
@app.route('/delete-report/<int:report_id>', methods=['POST'])
@admin_only
def delete_report(report_id):
    rpt = Report.query.get_or_404(report_id)
    ClaimRequest.query.filter_by(report_id=report_id).delete()
    Complaint.query.filter_by(report_id=report_id).delete()
    db.session.delete(rpt)
    db.session.commit()
    return ('', 204)

@app.route('/requests')
@admin_only
def view_requests():

    finalize_expired_claims()

    pending = ClaimRequest.query.filter_by(status='pending').all()
        # ── Compute waiting time for each claim ──
    now = datetime.utcnow()
    for cr in pending:
        delta = now - cr.timestamp
        days = delta.days
        hours = (delta.seconds // 3600)
        minutes = (delta.seconds % 3600) // 60

        if days:
            cr.wait_time = f"{days}d {hours}h {minutes}m"
        elif hours:
            cr.wait_time = f"{hours}h {minutes}m"
        else:
            cr.wait_time = f"{minutes}m"


    grouped = {}

    for cr in pending:
                # Grab the parent report
        rpt = cr.report

        if rpt.is_container:
            # Container-content match (with entropy bonus)
            content_rate = compute_content_match(
                rpt.container_contents or '',
                cr.description or ''
            )
            cr.match_percentage = round(content_rate * 100)
        else:
            # Description similarity
            desc_match = round(
                cosine_sim(
                    get_text_vec(rpt.description or ''),
                    get_text_vec(cr.description or '')
                ) * 100
            )
            # Location guesses similarity (take highest)
            loc_scores = [
                round(
                    cosine_sim(
                        get_text_vec(rpt.location or ''),
                        get_text_vec(cr.location1 or '')
                    ) * 100
                ),
                round(
                    cosine_sim(
                        get_text_vec(rpt.location or ''),
                        get_text_vec(cr.location2 or '')
                    ) * 100
                )
            ]
            cr.match_percentage = round((desc_match + max(loc_scores)) / 2)



        grouped.setdefault(cr.report_id, []).append(cr)

    # Sort each group by highest match %
    for key in grouped:
        grouped[key].sort(key=lambda r: r.match_percentage, reverse=True)

    # ── Inject accepted complaint if it exists ──
    for report_id, reqs in grouped.items():
        comp = Complaint.query.filter_by(
            report_id=report_id,
            status='accepted'
        ).first()
        if comp:
            # so template can render comp.user_email just like cr.user_email
            comp.user_email = comp.user.email if comp.user else ''
            # set a dummy match_percentage so it sorts consistently
            comp.match_percentage = 0
            # append into the same list
            reqs.append(comp)

    return render_template(
        'requests.html',
        grouped_requests=grouped,
        now=datetime.utcnow()  # 🆕 Add this line
    )


from datetime import timedelta

@app.route('/requests/<int:req_id>/<decision>', methods=['POST'])
@admin_only
def decide_claim(req_id, decision):
    cr = ClaimRequest.query.get_or_404(req_id)
    # Record which decision was chosen
    cr.decision = 'accept' if decision == 'accept' else 'decline'
    # Enter pending-finalization state
    cr.status = f"{cr.decision}_pending"
    cr.finalized = False
    cr.pending_until = datetime.utcnow() + timedelta(minutes=10)
    # Mark this claim as successful or not
    cr.is_successful = (decision == 'accept')

    db.session.commit()
    flash(f"Claim {cr.decision} pending. You have 10 minutes to undo.", "info")
    return redirect(url_for('view_requests'))


from datetime import datetime

@app.route('/requests/<int:req_id>/undo', methods=['POST'])
@admin_only
def undo_claim_decision(req_id):
    cr = ClaimRequest.query.get_or_404(req_id)
    # Only allow undo if still within the pending window
    if cr.finalized or not cr.pending_until or datetime.utcnow() > cr.pending_until:
        flash("Undo window has passed.", "warning")
    else:
        cr.status = 'pending'
        cr.decision = None
        cr.finalized = False
        cr.pending_until = None
        db.session.commit()
        flash("Decision undone. You can accept or decline again.", "info")
    return redirect(url_for('view_requests'))

@app.route('/requests/<int:req_id>/ask-in-person', methods=['POST'])
@admin_only
def ask_in_person(req_id):
    # 1. Parse the JSON payload
    data = request.get_json()
    deadline_str = data.get('deadline')
    notes        = data.get('notes', '')

    # 2. Convert the deadline into a datetime
    deadline = datetime.fromisoformat(deadline_str)

    # 3. Load the claim request
    cr = ClaimRequest.query.get_or_404(req_id)

    # 4. Update in-person verification fields
    cr.in_person_required     = True
    cr.in_person_deadline     = deadline
    cr.in_person_notes        = notes
    cr.in_person_requested_by = current_user.email
    cr.in_person_requested_at = datetime.utcnow()
    cr.status                 = 'in_person_pending'

    # 5. Save to the database
    db.session.commit()

    # 6. Respond OK
    return jsonify(success=True)

@app.route('/requests/<int:req_id>/verify-in-person', methods=['POST'])
@admin_only
def verify_in_person(req_id):
    cr = ClaimRequest.query.get_or_404(req_id)
    cr.in_person_verified = True
    cr.status = 'in_person_verified'
    db.session.commit()
    return ('', 204)


# New: “My Claims” for Bob (and any user)
@app.route('/my-claims')
@login_required
def my_claims():
    my_requests = ClaimRequest.query.\
        filter_by(user_email=current_user.email).\
        order_by(ClaimRequest.timestamp.desc()).all()
    complaints = Complaint.query.\
    filter_by(user_id=current_user.id).\
    order_by(Complaint.created_at.desc()).all()

    return render_template('myclaims.html', my_requests=my_requests, complaints=complaints)

@app.route('/admin/search')
@admin_only
def admin_search():
    """Page for admins to search reports via text or image."""
    return render_template('admin_search.html',
        show_sidebar=False, # ← hide nav
        show_brand=False)

# ── Home / dashboard ─────────────────────────────────────────
@app.route('/')
@login_required
def show_home():
    found_count = Report.query.filter_by(received=True).count()
    claims_total = ClaimRequest.query.count()
    claims_resolved = ClaimRequest.query.filter_by(status='accepted').count()

    return render_template(
        'home.html',
        found_count = found_count,
        claims_total = claims_total,
        claims_resolved = claims_resolved
    )


@csrf.exempt
@app.route("/api/check_match", methods=["POST"])
def api_check_match():
    """
    AJAX helper:
    • expects multipart/form-data with keys: image=<file>, text=<str>
    • returns JSON: {"ok": bool, "score": float}
    """
    file = request.files.get("photos")
    descr = request.form.get("text", "")

    # basic guard clauses
    if not file or not descr:
        return jsonify({"ok": False, "error": "missing image or text"}), 400

    # Save image to a temp file so PIL can open it
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    file.save(tmp.name)
    tmp.close()

    # compute similarity
    try:
        score = cosine_sim(
            get_image_vec(tmp.name),
            get_text_vec(descr)
        )
    finally:
        # always remove the temp file
        os.unlink(tmp.name)

    ok = score >= AI_MATCH_THRESHOLD
    return jsonify({"ok": ok, "score": round(score, 3)})

@csrf.exempt
@app.route('/api/search', methods=['POST'])
@admin_only
def api_search():
    """
    Handle admin searches by text or image.
    Returns JSON: { matches: [ {id, score, filename, description}, … ] }
    """
    # 1. Grab inputs
    text = request.form.get('text', '').strip()
    image = request.files.get('image')

    if not text and not image:
        return jsonify({'error': 'Provide text or image'}), 400

    # 2. Compute embedding
    if image:
        # Save to temp file so our util can open it
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        image.save(tmp.name)
        tmp.close()
        try:
            vec = get_image_vec(tmp.name)
        finally:
            os.unlink(tmp.name)
    else:
        vec = get_text_vec(text)

    # 3. Run FAISS search
    k = 5
    raw = current_app.vs.search(vec, k)

    # 4. Build match list with metadata
    matches = []
    for rid, score in raw:
        rpt = Report.query.get(rid)
        if not rpt:
            continue
        matches.append({
            'id': rpt.id,
            'score': score,
            'filename': rpt.filename,
            'description': rpt.description
        })

    return jsonify({'matches': matches})

@app.route('/admin/run_fraud_check')
@login_required # if you have login protection
def run_fraud_check():
    # Import and run your existing abuse checker logic
    from fraud import get_flagged_users
    flagged = get_flagged_users() # returns list of (email, count, avg_score)
    return render_template('fraud_results.html', flagged=flagged)



@app.context_processor
def inject_globals():
    five_days_ago = datetime.utcnow() - timedelta(days=5)
    user_email = current_user.email if current_user.is_authenticated else None

    read_thank_ids = {
        r.thank_you_id for r in NotificationRead.query.filter_by(user_email=user_email) if user_email
        if r.thank_you_id is not None
    }
    read_announcement_ids = {
        r.announcement_id for r in NotificationRead.query.filter_by(user_email=user_email)
        if r.announcement_id is not None
    }

    unread_thank_yous = ThankYouMessage.query.filter(
        ThankYouMessage.created_at >= five_days_ago,
        ~ThankYouMessage.id.in_(read_thank_ids)
    ).all()

    unread_announcements = Announcement.query.filter(
        Announcement.created_at >= five_days_ago,
        ~Announcement.id.in_(read_announcement_ids)
    ).all()

    notifications = [
        {
            'type': 'thank',
            'text': f"{t.claimant_email} claimed item #{t.report_id}. Thanks to {t.finder_email}!",
            'time': t.created_at
        } for t in unread_thank_yous
    ] + [
        {
            'type': 'admin',
            'text': a.message,
            'time': a.created_at
        } for a in unread_announcements
    ]

    notifications.sort(key=lambda x: x['time'], reverse=True)

    return {
        'email': user_email, # This is now correct
        'roles': current_user.roles.split(',') if current_user.is_authenticated else [], # This is now correct
        'today': date.today(),
        'notifications': notifications,
        'unread_count': len(unread_thank_yous) + len(unread_announcements)
    }







@app.route('/ask_for_proof/<int:claim_id>', methods=['POST'])
@login_required
def ask_for_proof(claim_id):
    # 1. Find the claim
    claim = ClaimRequest.query.get_or_404(claim_id)
    # 2. Only admins can do this
    if not current_user.is_admin:
        flash("Not allowed.", "danger")
        return redirect(url_for('found_items'))
    # 3. Change status so user knows to upload proof
    claim.status = 'requires proof'
    # 4. Set a 24-hour deadline
    claim.proof_deadline = datetime.utcnow() + timedelta(hours=24)
    db.session.commit()
    # 5. Let the admin see confirmation
    flash("Proof requested from user. They have 24 hours.", "info")
    return redirect(url_for('found_items'))

@app.route('/upload_proof/<int:claim_id>', methods=['POST'])
@login_required
def upload_proof(claim_id):
    claim = ClaimRequest.query.get_or_404(claim_id)
    # only the claimant may upload
    if claim.user_email != current_user.email:
        flash("Not your claim.", "danger")
        return redirect(url_for('my_claims'))

    # 1. get form fields
    proof_type = request.form['proof_type']
    file = request.files['proof_file']

    # 2. save file
    filename = secure_filename(file.filename)
    folder = os.path.join(app.config['UPLOAD_FOLDER'], 'proofs')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    file.save(path)

    # 4a. If it’s a receipt → OCR and text‐matching
    if proof_type == 'receipt':
        # run OCR
        text = ocr_image_to_text(path)
        claim.ocr_text = text
        # compare to original report description
        report_desc = claim.report.description
        score = cosine_sim(get_text_vec(text), get_text_vec(report_desc))
        claim.proof_score = float(score)

    # 4b. If it’s an image → feature‐vector matching
    else: # proof_type == 'image'
        vec = get_image_vec(path)
        claim.image_features = vec
        report_vec = claim.report.img_emb if claim.report.img_emb is not None else []
        score = cosine_sim(vec, report_vec)

        claim.proof_score = float(score)


    # 3. record on claim
    claim.proof_type = proof_type
    # (OCR / feature-extract later)
    claim.status = 'proof submitted'
    db.session.commit()

    flash("Proof uploaded! Admin will review soon.", "success")
    return redirect(url_for('my_claims'))


from flask import Blueprint, request, flash, redirect, url_for
from sqlalchemy.exc import SQLAlchemyError
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from datetime import datetime

# Assuming these are defined in your app.py or accessible via extensions
# from app import db, SENDGRID_KEY
# from app import Complaint, ClaimRequest, OwnershipEvent
# from app import generate_qr

admin_bp = Blueprint('admin_bp', __name__)

@admin_bp.route("/admin/complaints/<int:comp_id>/<string:decision>", methods=["POST"])
@admin_only
def decide_complaint(comp_id, decision):
    comp = Complaint.query.get_or_404(comp_id)
    rpt = comp.report
    old_owner = rpt.claimed_by or comp.user.email
    new_owner = comp.user.email

    if decision == "accept":
        try:
            # start transaction
            with db.session.begin():
                # 1) mark complaint accepted
                comp.status = "accepted"

                # 2) remove any old claims
                ClaimRequest.query.filter_by(report_id=rpt.id).delete()

                # 3) create new claim record
                new_claim = ClaimRequest(
                    report_id=rpt.id,
                    user_email=new_owner,
                    status="accepted",
                    timestamp=datetime.utcnow()
                )
                db.session.add(new_claim)

                # 4) log the transfer
                evt = OwnershipEvent(
                    report_id=rpt.id,
                    from_email=old_owner,
                    to_email=new_owner,
                    reason="Complaint accepted"
                )
                db.session.add(evt)

                # 5) update report ownership and QR
                rpt.claimed = True
                rpt.claimed_by = new_owner
                rpt.qr_code = generate_qr(f"Item #{rpt.id} now owned by {new_owner}")

                # 6) send notification email
                sg = SendGridAPIClient(SENDGRID_KEY)
                msg = Mail(
                    from_email='no-reply@yourdomain.com',
                    to_emails=old_owner,
                    subject=f"Ownership Transferred for Item #{rpt.id}",
                    html_content=(
                        f"Your claim on item #{rpt.id} has been revoked and "
                        f"transferred to {new_owner}."
                    )
                )
                resp = sg.send(msg)
                if resp.status_code >= 400:
                    raise Exception(f"Email send failed ({resp.status_code})")

            flash("Complaint accepted and original owner notified.", "success")

        except (SQLAlchemyError, Exception) as e:
            # rollback is automatic with `session.begin()` on exception
            flash(f"Transfer failed: {e}", "danger")

    else:
        # Decline path
        comp.status = "declined"
        db.session.commit()
        flash("Complaint declined.", "info")

    return redirect(url_for('admin_bp.admin_complaints'))

@admin_bp.route("/admin/complaints")
@admin_only
def admin_complaints():
    all_complaints = Complaint.query.order_by(Complaint.created_at.desc()).all()
    return render_template("admin_complaints.html", complaints=all_complaints)



from datetime import timedelta, date

@app.route('/lost-items')
@login_required
def lost_items():
    filter_date = request.args.get('filter_date')

    # ✅ Always show only approved posts to everyone (admin or not)
    q = LostItem.query.filter_by(received=True)


    if filter_date:
        q = q.filter_by(date_lost=filter_date)

    # Only keep recent posts (last 60 days)
    cutoff = datetime.utcnow() - relativedelta(months=2)
    q = q.filter(LostItem.timestamp >= cutoff)
    items = q.order_by(LostItem.timestamp.desc()).all()

    items_for_js = [
        {'id': i.id, 'description': i.description, 'location': i.location}
        for i in items
    ]

    max_date = date.today().isoformat()
    min_date = (date.today() - timedelta(days=60)).isoformat()

    return render_template(
        'lost_items.html',
        lost_posts=items,
        items_for_js=items_for_js,
        filter_date=filter_date,
        min_date=min_date,
        max_date=max_date,
        today=date.today()
    )


@app.route('/allow-lost-reports')
@admin_only
def allow_lost_reports():
    pending = LostItem.query.filter_by(received=False).order_by(LostItem.timestamp.desc()).all()
    return render_template('allow_lost_reports.html', pending=pending)

@app.route('/allow-lost-reports/<int:item_id>/accept', methods=['POST'])
@admin_only
def accept_lost_report(item_id):
    item = LostItem.query.get_or_404(item_id)
    item.received = True
    db.session.commit()
    flash("Lost item post accepted and published.", "success")
    return redirect(url_for('allow_lost_reports'))

@app.route('/allow-lost-reports/<int:item_id>/delete', methods=['POST'])
@admin_only
def delete_lost_report(item_id):
    item = LostItem.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    flash("Lost item post deleted.", "danger")
    return redirect(url_for('allow_lost_reports'))

@app.route('/lost-category/<cat>')
@login_required
def lost_category_items(cat):
    filter_date = request.args.get('filter_date')
    q = db.session.query(LostItem).filter(
        LostItem.received == True,
        func.lower(LostItem.category) == cat.lower()
    ).order_by(LostItem.timestamp.desc())

    if filter_date:
        q = q.filter_by(date_lost=filter_date)

    items = q.all()

    items_for_js = [
        {'id': i.id, 'description': i.description, 'location': i.location}
        for i in items
    ]

    user_email = current_user.email
    min_date = (date.today() - timedelta(days=60)).isoformat()
    max_date = date.today().isoformat()

    return render_template(
        'lost_category.html',  # you can reuse or clone `categoryitems.html`
        items=items,
        items_for_js=items_for_js,
        category=cat,
        email=user_email,
        roles=current_user.roles.split(','),
        filter_date=filter_date,
        min_date=min_date,
        max_date=max_date
    )

@app.route('/match-found-items/<int:lost_id>')
@login_required
def view_found_matches(lost_id):
    lost_item = LostItem.query.get_or_404(lost_id)

    if not lost_item.received:
        flash("This item hasn't been approved yet.", "warning")
        return redirect(url_for('lost_items'))

    # Get all accepted found reports
    found_reports = Report.query.filter_by(received=True).all()

    matches = []
    lost_vec = get_text_vec(lost_item.description + ' ' + lost_item.location)

    for report in found_reports:
        found_vec = get_text_vec(report.description + ' ' + report.location)
        score = float(cosine_sim(lost_vec, found_vec))
        matches.append({
            'id': report.id,
            'description': report.description,
            'location': report.location,
            'photo': report.photos[0].filename if report.photos else None,
            'score': score
        })

    top_matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:3]

    return render_template('found_matches.html', lost_item=lost_item, matches=top_matches)

@app.route("/admin/post-announcement", methods=["GET", "POST"])
@admin_only
def post_announcement():
    form = AnnouncementForm()
    if form.validate_on_submit():
        new_announcement = Announcement(message=form.message.data)
        db.session.add(new_announcement)
        db.session.commit()
        flash("Announcement posted!", "success")
        return redirect(url_for("post_announcement"))
    return render_template("post_announcement.html", form=form)


@app.route("/mark-notifications-read", methods=["POST"])
@login_required
def mark_notifications_read():
    email = current_user.email

    # Already read IDs
    seen_ann = {
        row.announcement_id
        for row in NotificationRead.query
        .filter_by(user_email=email)
        .filter(NotificationRead.announcement_id.isnot(None))
    }

    seen_thanks = {
        row.thank_you_id
        for row in NotificationRead.query
        .filter_by(user_email=email)
        .filter(NotificationRead.thank_you_id.isnot(None))
    }

    # New unseen
    new_ann = Announcement.query.filter(~Announcement.id.in_(seen_ann)).all()
    new_thanks = ThankYouMessage.query.filter(~ThankYouMessage.id.in_(seen_thanks)).all()

    for a in new_ann:
        db.session.add(NotificationRead(
            user_email=email,
            announcement_id=a.id
        ))

    for t in new_thanks:
        db.session.add(NotificationRead(
            user_email=email,
            thank_you_id=t.id
        ))

    db.session.commit()
    return jsonify(ok=True)

@app.route('/notify/<int:report_id>/<int:lost_id>', methods=['POST'])
@admin_only
def notify_owner(report_id, lost_id):
    found = Report.query.get_or_404(report_id)
    lost  = LostItem.query.get_or_404(lost_id)

    # Create an announcement for the lost‐item owner
    msg = (
        f"Hi {lost.user_email}, we found an item that might match your lost report "
        f"#{lost.id} (“{lost.description}”). Found details: {found.description} @ "
        f"{found.location} on {found.date_found}. "
        f"<a href='{url_for('found_items', _external=True)}'>Click to claim it</a>."
    )
    ann = Announcement(message=msg)
    db.session.add(ann)
    db.session.commit()

    flash(f"Sent in-app notification to {lost.user_email}.", "success")
    return redirect(url_for('allow_reports'))



# register our admin blueprint
app.register_blueprint(admin_bp)

# ─── App Init & Seeding ─────────────────────────────────────────────────────
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Seed default users
        for e, pw, roles in [
            ('alice@somaiya.edu','apple123','admin'),
            ('bob@somaiya.edu','banana@123',''),
            ('charles@somaiya.edu','cherry@123',''),
            ('daisy@somaiya.edu','desert@123','')
        ]:
            if not User.query.filter_by(email=e).first():
                db.session.add(User(email=e, password_hash=generate_password_hash(pw), roles=roles))
        db.session.commit()
    app.run(debug=True)