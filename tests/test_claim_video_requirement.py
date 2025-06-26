import pytest
from app import app, db, User, ClaimRequest

@pytest.fixture
def client(tmp_path):
    # Configure Flask for testing
    app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path/'test.db'}",
        "WTF_CSRF_ENABLED": False,
    })
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.drop_all()

def make_user(email, success_count, total_count):
    """
    Create a User with given email, and add ClaimRequest rows:
      - success_count successful claims (is_successful=True)
      - total_count - success_count failed claims (is_successful=False)
    """
    # Create and persist the user
    u = User(email=email, password_hash='irrelevant', roles='')  
    db.session.add(u)
    db.session.flush()  # assigns u.id
    
    # Seed successful claims
    for _ in range(success_count):
        db.session.add(ClaimRequest(user_id=u.id, report_id=1, is_successful=True))
    # Seed failed claims
    for _ in range(total_count - success_count):
        db.session.add(ClaimRequest(user_id=u.id, report_id=1, is_successful=False))
    
    db.session.commit()
    return u

def test_low_rate_requires_video(client):
    # 0️⃣ Seed a user with 0 successes out of 2 attempts → success_rate = 0.0
    with app.app_context():
        low_user = make_user('low@somaiya.edu', success_count=0, total_count=2)

    # 1️⃣ Log them in via your login endpoint (adjust field names as needed)
    client.post('/login', data={'email': low_user.email, 'password': 'irrelevant'})

    # 2️⃣ Attempt to claim without attaching a video
    resp = client.post('/claim/1', data={
        'description': 'Test claim',
        'location1': 'X',
        'location2': 'Y',
        # Note: no 'video_file' field here
    })

    # 3️⃣ We expect a 400 error and the 'upload a short video' message
    assert resp.status_code == 400
    data = resp.get_json()
    assert 'upload a short video' in data['message']
