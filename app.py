import os
from dotenv import load_dotenv
from datetime import date,timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, abort, jsonify
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, SelectField, DateField, FileField, TelField
from wtforms.validators import DataRequired, Length, Regexp
import bleach

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

csrf = CSRFProtect(app)
db = SQLAlchemy(app)

from difflib import SequenceMatcher

def similarity_score(text1, text2):
    return round(SequenceMatcher(None, text1.lower(), text2.lower()).ratio() * 100)

# ─── Models ─────────────────────────────────────────────────────────────────
class User(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String, unique=True, nullable=False)
    password_hash = db.Column(db.String, nullable=False)
    roles         = db.Column(db.String, nullable=False, default='')

class Report(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    email       = db.Column(db.String, nullable=False)
    filename    = db.Column(db.String, nullable=False)
    description = db.Column(db.String, nullable=False)
    location    = db.Column(db.String, nullable=False)
    date_found  = db.Column(db.String, nullable=False)  # 'YYYY-MM-DD'
    category    = db.Column(db.String, nullable=False)
    contact     = db.Column(db.String, nullable=False)
    timestamp   = db.Column(db.DateTime, server_default=db.func.now())
    claimed     = db.Column(db.Boolean, default=False)
    claimed_by  = db.Column(db.String, nullable=True)
    received    = db.Column(db.Boolean, default=False)

class ClaimRequest(db.Model):
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
    report      = db.relationship('Report', backref=db.backref('claim_requests', lazy=True))


class Complaint(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    reporter_email = db.Column(db.String, nullable=False)
    report_id      = db.Column(db.Integer, db.ForeignKey('report.id'), nullable=False)
    message        = db.Column(db.String, nullable=True)
    timestamp      = db.Column(db.DateTime, server_default=db.func.now())
    report         = db.relationship('Report', backref=db.backref('complaints', lazy=True))

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
    photo = FileField('Photo', validators=[DataRequired()])

# ─── Context Processor ──────────────────────────────────────────────────────
@app.context_processor
def inject_globals():
    return {
        'email': session.get('email'),
        'roles': session.get('roles', []),
        'today': date.today()
    }

# ─── Utility Decorators ─────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'email' not in session:
            return redirect(url_for('do_login'))
        return f(*args, **kwargs)
    return wrapper

def admin_only(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'email' not in session or 'admin' not in session.get('roles', []):
            abort(403)
        return f(*args, **kwargs)
    return wrapper

@app.route('/allow-reports')
@admin_only
def allow_reports():
    pending_reports=Report.query.filter_by(received=False).order_by(Report.timestamp.desc()).all()
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

# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
@login_required
def show_home():
    found_count = Report.query.filter_by(received=True).count()
    return render_template('home.html', found_count=found_count)

@app.route('/login', methods=['GET','POST'], endpoint='do_login')
def do_login():
    if request.method == 'POST':
        email = request.form['email']
        pw    = request.form['password']
        user  = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, pw):
            flash('Invalid credentials', 'danger')
            return redirect(url_for('do_login'))
        session['email'] = user.email
        session['roles'] = user.roles.split(',') if user.roles else []
        return redirect(url_for('show_home'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('do_login'))

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/report-found', methods=['GET','POST'])
@login_required
def report_found():
    form = ReportForm()
    if form.validate_on_submit():
        f = form.photo.data
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        desc = bleach.clean(form.description.data)
        rpt = Report(
            email       = session['email'],
            filename    = filename,
            description = desc,
            location    = form.location.data,
            date_found  = form.date_found.data.strftime('%Y-%m-%d'),
            category    = form.category.data,
            contact     = form.contact.data,
            received    = False
        )
        db.session.add(rpt)
        db.session.commit()
        flash('Report submitted!', 'success')
        return redirect(url_for('category_items', cat=form.category.data))
    return render_template('report_found.html', form=form)

@app.route('/found-items')
@login_required
def items_found():
    accepted = ClaimRequest.query.filter_by(status='accepted').all()
    return render_template('itemsfound.html', accepted=accepted)

@app.route('/category/<cat>')
@login_required
def category_items(cat):
    filter_date = request.args.get('filter_date')
    q = db.session.query(Report).filter(
        Report.category == cat,
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
        cr.report_id for cr in ClaimRequest.query.filter_by(user_email=session['email'])
    }

    return render_template(
        'categoryitems.html',
        items=items,
        items_for_js=items_for_js,
        category=cat,
        min_date=min_date,
        max_date=max_date,
        filter_date=filter_date,
        user_claims=user_claims
    )

# AJAX‐style Claim → JSON
@app.route('/claim/<int:report_id>', methods=['POST'])
@login_required
def claim_report(report_id):
    exists = ClaimRequest.query.filter_by(
        user_email=session['email'],
        report_id=report_id
    ).first()
    
    if not exists:
        cr = ClaimRequest(user_email=session['email'], report_id=report_id)
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
        desc_match = similarity_score(report_desc, claim_desc)

        # Location guesses similarity (take highest)
        loc1 = cr.location1 or ''
        loc2 = cr.location2 or ''
        loc_scores = [
            similarity_score(report_loc, loc1),
            similarity_score(report_loc, loc2),
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
        filter_by(user_email=session['email']).\
        order_by(ClaimRequest.timestamp.desc()).all()
    return render_template('myclaims.html', my_requests=my_requests)

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

if __name__ == '__main__':
    app.run(debug=True)

