import os
from dotenv import load_dotenv
from datetime import date,timedelta
from datetime import datetime

from wtforms.validators import Optional

from functools import wraps
import numpy as np
import pickle
import faiss
from flask import request, redirect, url_for, flash, render_template
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user
)
from flask_login import UserMixin


from werkzeug.security import check_password_hash


from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, abort, jsonify
)
# ... existing imports ...

from PIL import Image, ImageFilter


# --- AI helpers ---

import tempfile, os
from ai_utils import get_image_vec, get_text_vec, cosine_sim, ocr_image_to_text
# Tune this if you like: higher = stricter, lower = looser
AI_MATCH_THRESHOLD = 0.23
# If cosine similarity ≥ this, treat the photo as a duplicate
DUPLICATE_THRESHOLD = 0.90


from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from werkzeug.utils import secure_filename
from sqlalchemy import func  

from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import HiddenField, StringField, SelectField, DateField, FileField, SubmitField, TelField, TextAreaField
from wtforms.validators import DataRequired, Length, Regexp
import bleach

load_dotenv()

app = Flask(__name__)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)



from flask_login import current_user

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

from flask import request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        pw    = request.form.get('password')
        user  = User.query.filter_by(email=email).first()
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
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.config['QR_FOLDER'] = os.path.join('static', 'qr')
os.makedirs(app.config['QR_FOLDER'], exist_ok=True)


csrf = CSRFProtect(app)
db = SQLAlchemy(app)
migrate = Migrate(app, db)


from vector_store import VectorStore


# words that mark a report as urgent
URGENT_KEYWORDS = [
    "passport", "visa", "epipen", "epi-pen", "insulin", "inhaler",
    "medical", "prescription", "hearing aid", "oxygen", "oxygen tank",
]

import re

def compute_quality_score(text: str) -> float:
    """Return float 0–1 using same heuristic as JS."""
    text = text.lower().strip()
    words = text.split()
    length   = min(len(words) / 12, 0.4)
    has_col  = 0.3 if re.search(r'\b(black|white|red|blue|green|brown|grey)\b', text) else 0
    has_brand= 0.3 if re.search(r'\b(casio|hp|nike|titan|sony|apple|samsung|lenovo)\b', text) else 0
    has_num  = 0.2 if re.search(r'\b\d{3,}\b', text) else 0
    has_kw   = 0.2 if re.search(r'\b(wallet|phone|book|id|card|pen|earphones|calculator|bag)\b', text) else 0
    has_loc  = 0.2 if re.search(r'\b(library|canteen|lab|class|hall|ground)\b', text) else 0
    return min(length + has_col + has_brand + has_num + has_kw + has_loc, 1.0)


def compute_priority(description: str) -> float:
    """Return 1.0 if any urgent keyword appears, else 0.0."""
    desc_lc = description.lower()
    for w in URGENT_KEYWORDS:
        if w in desc_lc:
            return 1.0
    return 0.0

def predict_category(description):
    description = description.lower()
    if any(word in description for word in ["book", "novel", "pages", "notes"]):
        return "books"
    elif any(word in description for word in ["wallet", "purse", "card", "cash","bag","backpack","rucksack","tote"]):
        return "accessories"
    elif any(word in description for word in ["id", "pan", "license", "aadhar"]):
        return "identity"
    elif any(word in description for word in ["pen", "pencil", "sharpener"]):
        return "stationary"
    else:
        return "others"

import qrcode               #  ← make sure this import is near the others
from uuid import uuid4      # we’ll use this to give each QR a unique name

def generate_qr(data: str) -> str:
    """Create a PNG QR code containing *data*.
    Returns the **relative path** (e.g. 'qr/abc123.png') that you can pass to <img src>.
    """
    # 1. build the QR image in memory
    img = qrcode.make(data)

    # 2. choose a unique filename
    fname = f"{uuid4().hex}.png"
    rel_path = f"qr/{fname}"           # 'qr/....png'
    abs_path = os.path.join(app.config['QR_FOLDER'], fname)  # static/qr/....png

    # 3. save
    img.save(abs_path)

    print(f"QR generated → {rel_path}", flush=True)   # ← add this

    return rel_path
     # something we can embed as /static/qr/...

# ─── Models ─────────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String, unique=True, nullable=False)
    password_hash = db.Column(db.String, nullable=False)
    roles         = db.Column(db.String, nullable=False, default='')
    @property
    def is_admin(self):
        return 'admin' in self.roles.split(',')


class Report(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    email       = db.Column(db.String, nullable=False)
    filenames   = db.Column(db.PickleType, nullable=True)
    description = db.Column(db.String, nullable=False)
    location    = db.Column(db.String, nullable=False)
    date_found  = db.Column(db.String, nullable=False)  # 'YYYY-MM-DD'
    category    = db.Column(db.String, nullable=False)
    contact     = db.Column(db.String, nullable=False)
    timestamp   = db.Column(db.DateTime, server_default=db.func.now())
    claimed     = db.Column(db.Boolean, default=False)
    claimed_by  = db.Column(db.String, nullable=True)
    received    = db.Column(db.Boolean, default=False)
    img_emb = db.Column(db.PickleType)  # stores the image vector
    priority_score = db.Column(db.Float, default=0.0)
    qr_code      = db.Column(db.String, nullable=True)   # holds 'qr/<uuid>.png'
    matched_report_id = db.Column(db.Integer, nullable=True)  # ID of matched lost item
    notification_sent = db.Column(db.Boolean, default=False)  # Track if owner was notified



class ClaimRequest(db.Model):
    __table_args__ = (
        db.Index('ix_claim_req_report_created', 'report_id', 'created_at'),
        db.Index('ix_claim_req_user_email', 'user_email'),
    )
    id          = db.Column(db.Integer, primary_key=True)
    user_email  = db.Column(db.String, nullable=False)
    report_id   = db.Column(db.Integer, db.ForeignKey('report.id'), nullable=False)
    status      = db.Column(db.String, nullable=False, default='pending')
    timestamp   = db.Column(db.DateTime, server_default=db.func.now())
    report      = db.relationship('Report', backref=db.backref('claim_requests', lazy=True))
    description = db.Column(db.String)
    location1   = db.Column(db.String)
    location2   = db.Column(db.String)
    location3   = db.Column(db.String)
    quality_score = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    delay_seconds = db.Column(db.Float, nullable=True)
    proof_type     = db.Column(db.String(20), nullable=True)
    ocr_text       = db.Column(db.Text, nullable=True)
    image_features = db.Column(db.PickleType, nullable=True)
    proof_score    = db.Column(db.Float, nullable=True)



# ------------------------
# Complaint model (NEW)
# ------------------------
class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(
        db.Integer,
        db.ForeignKey('report.id', name='fk_complaint_report'),
        nullable=False
    )
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id', name='fk_complaint_user'),
        nullable=True
    )

    details = db.Column(db.Text, nullable=False)
    proof_filename = db.Column(db.String(255), nullable=True)
    quality_score = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='complaints', lazy=True)
    proof_type     = db.Column(db.String(20), nullable=True)
    ocr_text       = db.Column(db.Text, nullable=True)
    image_features = db.Column(db.PickleType, nullable=True)
    proof_score    = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), nullable=False, default='pending')





# ─── Forms ───────────────────────────────────────────────────────────────────
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
    photo = FileField('Photos', render_kw={"multiple": True}, validators=[DataRequired()])


class ReportFormLost(FlaskForm):
    description = StringField('Description', validators=[
        DataRequired(), Length(max=100),
        Regexp(r'^[A-Za-z0-9\s\.!\-]+$', message="Only letters, numbers, spaces, and .!- allowed.")
    ])
    location = StringField('Location', validators=[
        DataRequired(), Length(max=50),
        Regexp(r'^[A-Za-z0-9\s,\-]+$', message="Only letters, numbers, spaces, commas, dashes allowed.")
    ])
    date_lost = DateField('Date Lost', validators=[DataRequired()], format='%Y-%m-%d')
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
    photo = FileField('Photos(Optional)', validators=[Optional()])  # ✅ optional

# ------------------------
# ComplaintForm  (NEW)
# ------------------------
class ComplaintForm(FlaskForm):
    report_id = HiddenField()
    details = TextAreaField(
        "Why this item belongs to you",
        validators=[DataRequired(), Length(min=30)]
    )
    proof = FileField("Proof (image / PDF)")
    submit = SubmitField("Submit Complaint")


# ─── Context Processor ──────────────────────────────────────────────────────
@app.context_processor
def inject_globals():
    return {
        'email': session.get('email'),
        'roles': session.get('roles', []),
        'today': date.today()
    }

# ─── Utility Decorators ─────────────────────────────────────────────────────


from flask_login import current_user   # make sure this import is near the top

def admin_only(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return wrapper


@app.route('/allow-reports')
@admin_only
def allow_reports():
    view = request.args.get('view', 'found')  # default to 'found'

    query = Report.query.order_by(
        Report.priority_score.desc(),
        Report.timestamp.desc()
    )

    if view == 'found':
        reports = query.filter_by(received=False, claimed=False).all()
    elif view == 'lost':
        reports = query.filter_by(received=False, claimed=True).all()
    else:
        # fallback: avoid crash
        reports = []

    return render_template('allow_reports.html', reports=reports, view=view)


@app.route('/allow-report/<int:report_id>/accept', methods=['POST'])
@admin_only
def accept_report(report_id):
    rpt = Report.query.get_or_404(report_id)
    
    # NEW CHECK: Require notification for matched items
    if rpt.matched_report_id and not rpt.notification_sent:
        return jsonify(error="Must notify owner before approval"), 400
    
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
@app.route("/complain/<int:report_id>", methods=["GET", "POST"])
@login_required               # ← keep if you already use Flask-Login
def complain(report_id):
    report = Report.query.get_or_404(report_id)

    form = ComplaintForm()
    form.report_id.data = report_id  # hidden field

    if form.validate_on_submit():
        # --- 1. save uploaded proof file (if any) ---
        proof_file = form.proof.data
        proof_filename = None
        if proof_file:
            proof_filename = secure_filename(proof_file.filename)
            proof_path = os.path.join(
                app.config["UPLOAD_FOLDER"], "complaints", proof_filename
            )
            os.makedirs(os.path.dirname(proof_path), exist_ok=True)
            proof_file.save(proof_path)
    
    # AI scoring of Daisy’s proof
    if proof_filename and form.proof.data:
        if form.proof.data.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # image proof
            vec = get_image_vec(path)
            complaint.image_features = vec
            complaint.proof_type = 'image'
            # compare to report img_emb
            rpt_vec = complaint.report.img_emb or []
            complaint.proof_score = cosine_sim(vec, rpt_vec)
        else:
            # assume receipt → OCR
            txt = ocr_image_to_text(path)
            complaint.ocr_text = txt
            complaint.proof_type = 'receipt'
            # compare to report.description
            complaint.proof_score = cosine_sim(get_text_vec(txt),
                                            get_text_vec(complaint.report.description))
# then db.session.add(complaint) & commit as before


        # --- 2. VERY simple AI score placeholder ---
        # (replace later with fancy NLP)
        txt_len = len(form.details.data.strip())
        quality_score = min(txt_len / 200.0, 1.0)  # 0 - 1 scale

        # --- 3. create & store complaint ---
        complaint = Complaint(
            report_id=report.id,
            user_id=current_user.id,  # remove if no user system
            details=form.details.data,
            proof_filename=proof_filename,
            quality_score=quality_score,
        )
        db.session.add(complaint)
        db.session.commit()

        flash("Complaint submitted ✓ Admins will review soon.", "success")
        return redirect(url_for("show_home"))  # change target as you like

    return render_template("complaint_form.html", form=form)


# ─── Routes ─────────────────────────────────────────────────────────────────




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

@app.route('/report-lost', methods=['GET', 'POST'])
@login_required
def report_lost():
    form = ReportFormLost()

    session.pop('_flashes', None)

    if request.method == 'POST' and form.validate_on_submit():
        desc = bleach.clean(form.description.data)
        category = predict_category(desc)
        if form.category.data != category:
            flash(f"This item was stored under '{category.title()}' instead of '{form.category.data.title()}' because it matched better.")
            form.category.data = category

        filenames = []
        uploaded_files = request.files.getlist('photo')
        for f in uploaded_files:
            if f and f.filename:
                filename = secure_filename(f.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(path)
                filenames.append(filename)

        img_vec = None
        if filenames:
            try:
                img_vec = get_image_vec(os.path.join(app.config['UPLOAD_FOLDER'], filenames[0]))
            except:
                pass  # skip embedding if failed

        rpt = Report(
            email       = current_user.email,
            filenames   = filenames,
            description = desc,
            location    = form.location.data,
            date_found  = form.date_lost.data.strftime('%Y-%m-%d'),  # using date_found field
            category    = form.category.data,
            contact     = form.contact.data,
            claimed     = True,     # LOST implies already claimed
            received    = False,    # Awaiting admin approval
            img_emb     = img_vec,
            priority_score = compute_priority(desc)
        )
        db.session.add(rpt)
        db.session.commit()

        flash('Lost item reported! Awaiting admin approval.', 'success')
        return redirect(url_for('report_lost'))

    return render_template('report_lost.html', form=form)


@app.route('/report-found', methods=['GET','POST'])
@csrf.exempt
@login_required
def report_found():
    form = ReportForm()
    if form.validate_on_submit():
        predicted = predict_category(form.description.data)
        if form.category.data != predicted:
            flash(f"This item was stored under '{predicted.title()}' instead of '{form.category.data.title()}' because it matched better.")
            form.category.data = predicted

        uploaded_files = request.files.getlist('photo')
        filenames = []

        for f in uploaded_files:
            if f and f.filename:
                filename = secure_filename(f.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(save_path)

                # Blur and save as blurred_<filename>
                img = Image.open(save_path)
                blurred = img.filter(ImageFilter.GaussianBlur(radius=10))
                blurred.save(os.path.join(app.config['UPLOAD_FOLDER'], 'blurred_' + filename))

                filenames.append(filename)

        # --- Compute image embedding ---
        img_vec = None
        if filenames:
            try:
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], filenames[0])
                img_vec = get_image_vec(img_path)
            except Exception as e:
                print(f"Error computing image vector: {e}")
                img_vec = None

        desc = bleach.clean(form.description.data)
        
        # ---- DESCRIPTION-vs-PHOTO GUARD ----
        if img_vec is not None:
            text_vec = get_text_vec(desc)
            similarity = cosine_sim(img_vec, text_vec)
            print(f"Mismatch-check sim={similarity:.3f}")
            if similarity < AI_MATCH_THRESHOLD:
                flash("The photo doesn't seem to match the description. Please revise.", "warning")
                return redirect(url_for('report_found'))
        # ---- END GUARD ----

        # ---- FAISS-backed duplicate check ----
        if img_vec is not None and vs.id_map:
            # Find up to 3 nearest neighbors by cosine similarity
            dups = vs.search(img_vec, k=3)
            if dups:
                top_id, top_score = dups[0]  # best match
                print(f"FAISS-dup-check: against #{top_id}  score={top_score:.3f}")
                if top_score >= DUPLICATE_THRESHOLD:
                    flash(f"It looks like this item has already been reported (score: {top_score:.2f}).", "warning")
                    return redirect(url_for('report_found'))

        # Create and save the new report
        rpt = Report(
            email = current_user.email,
            filenames = filenames,
            description = desc,
            location = form.location.data,
            date_found = form.date_found.data.strftime('%Y-%m-%d'),
            category = form.category.data,
            contact = form.contact.data,
            received = False,
            img_emb = img_vec,
            priority_score = compute_priority(desc),
            matched_report_id = None  # Initialize as None
        )
        db.session.add(rpt)
        db.session.commit()

        # Update FAISS index
        if img_vec is not None:
            vec = np.array([img_vec], dtype=np.float32)
            faiss.normalize_L2(vec)
            try:
                vs.index.add(vec)
            except AssertionError:
                # Rebuild the FAISS index for the correct dimension
                dim = vec.shape[1]
                vs.index = faiss.IndexFlatIP(dim)
                vs.index.add(vec)
            vs.id_map.append(rpt.id)
            # Persist updated index + ID map
            faiss.write_index(vs.index, vs.index_path)
            with open(vs.map_path, 'wb') as f:
                pickle.dump(vs.id_map, f)

        # --- AUTOMATIC MATCHING WITH LOST ITEMS ---
        # Get all approved lost items (claimed and approved by admin)
        lost_items = Report.query.filter_by(claimed=True, received=True).all()
        
        best_match = None
        highest_score = 0
        MATCH_THRESHOLD = 0.7  # Similarity threshold for a match
        
        for lost_item in lost_items:
            # Compare descriptions using AI text vectors
            found_vec = get_text_vec(rpt.description)
            lost_vec = get_text_vec(lost_item.description)
            similarity = cosine_sim(found_vec, lost_vec)
            
            # Check if this is the best match above threshold
            if similarity > highest_score and similarity >= MATCH_THRESHOLD:
                highest_score = similarity
                best_match = lost_item

        # Save the best match if found
        if best_match:
            rpt.matched_report_id = best_match.id
            db.session.commit()
            flash(f'Report submitted! Potential match found with lost item #{best_match.id}', 'success')
        else:
            flash('Report submitted!', 'success')
            
        return redirect(url_for('report_found', cat=form.category.data))
    
    return render_template('report_found.html', form=form)

from sqlalchemy import or_

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
    view_type = request.args.get('view', 'found')  # default to found

    query = db.session.query(Report).filter(
        func.lower(Report.category) == cat.lower()
    )

    if view_type == 'found':
        query = query.filter(Report.received == True, Report.claimed == False)
    elif view_type == 'lost' and current_user.is_admin:
        query = query.filter(Report.received == True, Report.claimed == True)

    if filter_date:
        query = query.filter_by(date_found=filter_date)

    items = query.order_by(Report.timestamp.desc()).all()

    items_for_js = [
        {'id': i.id, 'description': i.description, 'location': i.location}
        for i in items
    ]

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
        view=view_type
    )

# AJAX‐style Claim → JSON
@app.route('/claim/<int:report_id>', methods=['POST'])
@login_required
def claim_report(report_id):
    exists = ClaimRequest.query.filter_by(
        user_email=current_user.email,
        report_id=report_id
    ).first()
    
    
    if not exists:
            reason = request.form.get('description', '')
            q_score = compute_quality_score(reason)
            report = Report.query.get(report_id)
            delay = (datetime.utcnow() - report.timestamp).total_seconds()


            cr = ClaimRequest(
                    report_id   = report_id,
                    user_email  = current_user.email,
                    description      = reason,
                    quality_score = q_score,    # <-- NEW
                    created_at     = datetime.utcnow(),
                    delay_seconds  = delay
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
    pending = ClaimRequest.query.filter_by(status='pending').all()
    grouped = {}

    for cr in pending:
        report_desc = cr.report.description or ''
        report_loc  = cr.report.location or ''
        claim_desc  = cr.description or ''

        # Description similarity
        desc_match = round(
            cosine_sim(get_text_vec(report_desc), get_text_vec(claim_desc)) * 100
        )


        # Location guesses similarity (take highest)
        loc1 = cr.location1 or ''
        loc2 = cr.location2 or ''
        loc_scores = [
            round(cosine_sim(get_text_vec(report_loc), get_text_vec(loc1)) * 100),
            round(cosine_sim(get_text_vec(report_loc), get_text_vec(loc2)) * 100),
        ]

        best_location_match = max(loc_scores)

        # Final score = average of both
        final_match = round((desc_match + best_location_match) / 2)

        

        # Attach for display
        cr.desc_match = desc_match
        cr.loc_match = best_location_match
        cr.match_percentage = final_match

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

    return render_template('requests.html', grouped_requests=grouped)

@app.route('/requests/<int:req_id>/<decision>', methods=['POST'])
@admin_only
def decide_claim(req_id, decision):
    cr = ClaimRequest.query.get_or_404(req_id)
    if decision == 'accept':
        cr.status = 'accepted'
        rpt = cr.report
        rpt.claimed    = True
        rpt.claimed_by = cr.user_email
                # ---- QR GENERATION ---------------------------------
        qr_rel_path = generate_qr(f"Found item #{rpt.id} claimed by {cr.user_email}")
        rpt.qr_code = qr_rel_path         # store path on the Report
        # -----------------------------------------------------

        others = ClaimRequest.query.filter(
            ClaimRequest.report_id==cr.report_id,
            ClaimRequest.id!=req_id
        )
        for o in others:
            o.status = 'declined'
        db.session.commit()
        flash('Claim accepted.', 'success')
    else:
        cr.status = 'declined'
        db.session.commit()
        flash('Claim declined.', 'info')
    return redirect(url_for('view_requests'))

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
        show_sidebar=False,   #  ← hide nav
        show_brand=False)

# ── Home / dashboard ─────────────────────────────────────────
@app.route('/')
@login_required
def show_home():
    found_count      = Report.query.filter_by(received=True).count()
    claims_total     = ClaimRequest.query.count()
    claims_resolved  = ClaimRequest.query.filter_by(status='accepted').count()

    return render_template(
        'home.html',
        found_count      = found_count,
        claims_total     = claims_total,
        claims_resolved  = claims_resolved
    )



# ─── Init & Seed ────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()
    if not User.query.filter_by(email='alice@somaiya.edu').first():
        db.session.add(User(
            email='alice@somaiya.edu',
            password_hash=generate_password_hash('apple123'),
            roles='admin'
        ))
    if not User.query.filter_by(email='bob@somaiya.edu').first():
        db.session.add(User(
            email='bob@somaiya.edu',
            password_hash=generate_password_hash('banana@123'),
            roles=''
        ))
    if not User.query.filter_by(email='charles@somaiya.edu').first():
        db.session.add(User(
            email='charles@somaiya.edu',
            password_hash=generate_password_hash('cherry@123'),
            roles=''
        ))
    if not User.query.filter_by(email='daisy@somaiya.edu').first():
        db.session.add(User(
            email='daisy@somaiya.edu',
            password_hash=generate_password_hash('desert@123'),
            roles=''
        ))
    db.session.commit()
vs = VectorStore()

@csrf.exempt
@app.route("/api/check_match", methods=["POST"])
def api_check_match():
    """
    AJAX helper:
    • expects multipart/form-data with keys: image=<file>, text=<str>
    • returns JSON: {"ok": bool, "score": float}
    """
    file = request.files.get("image")
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
    raw = vs.search(vec, k)

    # 4. Build match list with metadata
    matches = []
    for rid, score in raw:
        rpt = Report.query.get(rid)
        if not rpt:
            continue
        matches.append({
            'id':          rpt.id,
            'score':       score,
            'filename':    rpt.filename,
            'description': rpt.description
        })

    return jsonify({'matches': matches})

@app.route('/admin/run_fraud_check')
@login_required  # if you have login protection
def run_fraud_check():
    # Import and run your existing abuse checker logic
    from fraud import get_flagged_users
    flagged = get_flagged_users()  # returns list of (email, count, avg_score)
    return render_template('fraud_results.html', flagged=flagged)

@app.context_processor
def inject_current_user():
    return dict(current_user=current_user)

# --------------------------------------------------
# Admin: view all ownership complaints
# URL:   /admin/complaints
# --------------------------------------------------
@app.route("/admin/complaints")
@admin_only
def admin_complaints():
    complaints = (
        Complaint.query.order_by(Complaint.created_at.desc()).all()
    )
    

    return render_template(
        "admin_complaints.html",
        complaints=complaints,
    )

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
    else:  # proof_type == 'image'
        vec = get_image_vec(path)
        claim.image_features = vec
        report_vec = claim.report.img_emb or []
        score = cosine_sim(vec, report_vec)
        claim.proof_score = float(score)

    
    # 3. record on claim
    claim.proof_type = proof_type
    # (OCR / feature-extract later)
    claim.status = 'proof submitted'
    db.session.commit()
    
    flash("Proof uploaded! Admin will review soon.", "success")
    return redirect(url_for('my_claims'))


@app.route("/admin/complaints/<int:comp_id>/<string:decision>", methods=["POST"])
@admin_only
def decide_complaint(comp_id, decision):
    comp = Complaint.query.get_or_404(comp_id)
    if decision == "accept":
        comp.status = "accepted"
        flash("Complaint accepted.", "success")
    else:
        comp.status = "declined"
        flash("Complaint declined.", "info")
    db.session.commit()
    return redirect(url_for("admin_complaints"))

@app.context_processor
def inject_models():
    return dict(Report=Report)

@app.route('/notify-owner/<int:report_id>', methods=['POST'])
@admin_only
def notify_owner(report_id):
    found_report = Report.query.get_or_404(report_id)
    if not found_report.matched_report_id:
        return jsonify(success=False, error="No matched lost item"), 400
    
    lost_report = Report.query.get_or_404(found_report.matched_report_id)
    
    # In real app, you would send an email here
    print(f"Notifying {lost_report.email} about potential match for item {lost_report.id}")
    
    # Update status
    found_report.notification_sent = True
    db.session.commit()
    
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)

