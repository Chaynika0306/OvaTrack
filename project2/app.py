from flask import Flask, render_template, request, session, redirect, url_for
import joblib
import numpy as np
import sqlite3
import hashlib
from datetime import datetime, timedelta
from functools import wraps

app = Flask(__name__)
app.secret_key = 'ovatrack_secret_2025'

# ── Load ML models ──────────────────────────────────────────────────────────────
pcos_model    = joblib.load("RandomForest_PCOS.pkl")
scaler        = joblib.load("scaler.pkl")
cycle_model   = joblib.load("cycle_model.pkl")
cycle_imputer = joblib.load("cycle_imputer.pkl")

FEATURE_NAMES = ['Age','Weight','Height','BMI','Cycle Length',
                 'Hair Growth','Acne','Hair Thinning','Exercise','Fast Food']

# ── Database ─────────────────────────────────────────────────────────────────────
DB = "ovatrack.db"

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT NOT NULL,
            email    TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created  TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            type       TEXT NOT NULL,
            risk       REAL,
            risk_level TEXT,
            bmi        REAL,
            cycle_days REAL,
            result     TEXT,
            created    TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS symptoms (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            log_date   TEXT NOT NULL,
            mood       INTEGER,
            pain_level INTEGER,
            acne       INTEGER DEFAULT 0,
            bloating   INTEGER DEFAULT 0,
            fatigue    INTEGER DEFAULT 0,
            headache   INTEGER DEFAULT 0,
            notes      TEXT,
            created    TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS cycle_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            last_period  TEXT NOT NULL,
            cycle_length INTEGER NOT NULL,
            next_period  TEXT,
            created      TEXT DEFAULT (datetime('now'))
        );
        """)

init_db()

# ── Helpers ───────────────────────────────────────────────────────────────────────
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def get_notification(uid):
    with get_db() as db:
        cycle = db.execute(
            "SELECT * FROM cycle_log WHERE user_id=? ORDER BY created DESC LIMIT 1", (uid,)
        ).fetchone()
    if not cycle or not cycle['next_period']:
        return None
    today  = datetime.today().date()
    next_p = datetime.strptime(cycle['next_period'], '%Y-%m-%d').date()
    delta  = (next_p - today).days
    if delta < 0:
        return {'type':'overdue','days':abs(delta),
                'message':f"Period was due {abs(delta)} day(s) ago.",'date':cycle['next_period']}
    elif delta <= 3:
        return {'type':'soon','days':delta,
                'message':f"Period expected in {delta} day(s).",'date':cycle['next_period']}
    elif delta <= 14:
        return {'type':'upcoming','days':delta,
                'message':f"Next period in {delta} days.",'date':cycle['next_period']}
    return {'type':'normal','days':delta,
            'message':f"Next period on {cycle['next_period']}.",'date':cycle['next_period']}

# ── AUTH ──────────────────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET','POST'])
def register():
    error = None
    if request.method == 'POST':
        name  = request.form['name'].strip()
        email = request.form['email'].strip().lower()
        pw    = request.form['password']
        if len(pw) < 6:
            error = "Password must be at least 6 characters."
        else:
            try:
                with get_db() as db:
                    db.execute("INSERT INTO users (name,email,password) VALUES (?,?,?)",
                               (name, email, hash_pw(pw)))
                return redirect(url_for('login', registered=1))
            except sqlite3.IntegrityError:
                error = "An account with that email already exists."
    return render_template('register.html', error=error)

@app.route('/login', methods=['GET','POST'])
def login():
    error = None
    registered = request.args.get('registered')
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        pw    = request.form['password']
        with get_db() as db:
            user = db.execute(
                "SELECT * FROM users WHERE email=? AND password=?",
                (email, hash_pw(pw))
            ).fetchone()
        if user:
            session['user_id']   = user['id']
            session['user_name'] = user['name']
            return redirect(url_for('home'))
        error = "Invalid email or password."
    return render_template('login.html', error=error, registered=registered)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ── HOME ──────────────────────────────────────────────────────────────────────────
@app.route('/')
@login_required
def home():
    return render_template('index.html', user_name=session.get('user_name',''))

# ── PCOS PREDICTION ───────────────────────────────────────────────────────────────
@app.route('/predict_pcos', methods=['POST'])
@login_required
def predict_pcos():
    try:
        age          = float(request.form['age'])
        weight       = float(request.form['weight'])
        height       = float(request.form['height'])
        cycle_length = float(request.form['cycle_length'])
        hirsutism    = int(request.form['hirsutism'])
        acne         = int(request.form['acne'])
        hair_thinning= int(request.form['hair_thinning'])
        fast_food    = int(request.form['fast_food'])
        exercise     = int(request.form['exercise'])
        last_period  = request.form.get('last_period','')

        if age < 9:
            return render_template('index.html',
                                   error="Age must be 9 or above.",
                                   user_name=session.get('user_name',''))

        bmi      = round(weight / ((height / 100) ** 2), 2)
        features = np.array([[age, weight, height, bmi, cycle_length,
                               hirsutism, acne, hair_thinning, exercise, fast_food]])
        scaled   = scaler.transform(features)

        risk        = round(pcos_model.predict_proba(scaled)[0][1] * 100, 2)
        pred        = pcos_model.predict(scaled)[0]
        importances = pcos_model.feature_importances_.tolist()

        if pred == 1:
            result_text = "PCOS Detected"
            suggestions = ["Adopt a balanced, low-glycemic diet.",
                           "Exercise regularly and maintain a healthy weight.",
                           "Prioritize sleep and reduce stress.",
                           "Consult a gynecologist or endocrinologist.",
                           "Consider regular tracking of cycles and symptoms."]
            risk_level  = "High" if risk >= 70 else "Medium"
        else:
            result_text = "No PCOS Detected"
            suggestions = ["Maintain your current healthy routine.",
                           "Keep exercising regularly.",
                           "Stay hydrated and eat balanced meals."]
            risk_level  = "Low"

        uid = session['user_id']
        with get_db() as db:
            db.execute("INSERT INTO predictions (user_id,type,risk,risk_level,bmi,result) VALUES (?,?,?,?,?,?)",
                       (uid, 'pcos', risk, risk_level, bmi, result_text))
            if last_period:
                next_dt = datetime.strptime(last_period,'%Y-%m-%d') + timedelta(days=int(cycle_length))
                db.execute("INSERT INTO cycle_log (user_id,last_period,cycle_length,next_period) VALUES (?,?,?,?)",
                           (uid, last_period, int(cycle_length), next_dt.strftime('%Y-%m-%d')))

        session['dash'] = {
            'type':'pcos', 'result_text':result_text,
            'risk':risk, 'risk_level':risk_level,
            'bmi':bmi, 'age':age, 'weight':weight, 'height':height,
            'cycle_length':cycle_length, 'suggestions':suggestions,
            'feature_names':FEATURE_NAMES, 'importances':importances
        }
        return redirect(url_for('dashboard'))

    except Exception as e:
        return render_template('index.html', error=str(e),
                               user_name=session.get('user_name',''))

# ── CYCLE PREDICTION ──────────────────────────────────────────────────────────────
@app.route('/predict_cycle', methods=['POST'])
@login_required
def predict_cycle():
    try:
        age          = float(request.form['age'])
        cycle_number = float(request.form['cycle_number'])
        conception   = 1 if request.form['conception_cycle'].lower() == 'yes' else 0
        last_period  = request.form.get('last_period','')

        if age < 9:
            return render_template('index.html',
                                   error="Age must be 9 or above.",
                                   user_name=session.get('user_name',''))

        feats      = np.array([[age, cycle_number, conception]])
        imputed    = cycle_imputer.transform(feats)
        cycle_days = round(float(cycle_model.predict(imputed)[0]), 1)

        if 21 <= cycle_days <= 35:
            status, note = "Normal","Predicted cycle is within the normal 21–35 day range."
        elif cycle_days < 21:
            status, note = "Short","Predicted cycle is shorter than average. Track for a few months."
        else:
            status, note = "Long","Predicted cycle is longer than average. Consider consulting a doctor."

        uid = session['user_id']
        with get_db() as db:
            db.execute("INSERT INTO predictions (user_id,type,cycle_days,result) VALUES (?,?,?,?)",
                       (uid, 'cycle', cycle_days, f"Predicted cycle: {cycle_days} days"))
            if last_period:
                next_dt = datetime.strptime(last_period,'%Y-%m-%d') + timedelta(days=int(cycle_days))
                db.execute("INSERT INTO cycle_log (user_id,last_period,cycle_length,next_period) VALUES (?,?,?,?)",
                           (uid, last_period, int(cycle_days), next_dt.strftime('%Y-%m-%d')))

        session['dash'] = {
            'type':'cycle',
            'result_text':f"Predicted cycle length: {cycle_days} days",
            'risk':None, 'risk_level':None,
            'cycle_days':cycle_days, 'cycle_status':status, 'cycle_note':note,
            'age':age,
            'suggestions':["Track your cycle consistently each month.",
                           "Note any symptoms like cramps or mood changes.",
                           "Maintain a healthy diet and sleep schedule."]
        }
        return redirect(url_for('dashboard'))

    except Exception as e:
        return render_template('index.html', error=str(e),
                               user_name=session.get('user_name',''))

# ── DASHBOARD ─────────────────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    data = session.get('dash')
    if not data:
        return redirect(url_for('home'))

    uid = session['user_id']
    with get_db() as db:
        symptoms     = db.execute(
            "SELECT * FROM symptoms WHERE user_id=? ORDER BY log_date DESC LIMIT 7",(uid,)).fetchall()
        last_cycle   = db.execute(
            "SELECT * FROM cycle_log WHERE user_id=? ORDER BY created DESC LIMIT 1",(uid,)).fetchone()
        recent_preds = db.execute(
            "SELECT * FROM predictions WHERE user_id=? ORDER BY created DESC LIMIT 5",(uid,)).fetchall()

    notif = get_notification(uid)
    return render_template('dashboard.html',
                           data=data, symptoms=symptoms,
                           last_cycle=last_cycle, recent_preds=recent_preds,
                           notification=notif,
                           user_name=session.get('user_name',''))

# ── SYMPTOMS ──────────────────────────────────────────────────────────────────────
@app.route('/symptoms', methods=['GET','POST'])
@login_required
def symptoms():
    uid     = session['user_id']
    message = None
    if request.method == 'POST':
        with get_db() as db:
            db.execute("""INSERT INTO symptoms
                (user_id,log_date,mood,pain_level,acne,bloating,fatigue,headache,notes)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (uid,
                 request.form.get('log_date', datetime.today().strftime('%Y-%m-%d')),
                 int(request.form.get('mood', 3)),
                 int(request.form.get('pain_level', 0)),
                 1 if request.form.get('acne')     else 0,
                 1 if request.form.get('bloating') else 0,
                 1 if request.form.get('fatigue')  else 0,
                 1 if request.form.get('headache') else 0,
                 request.form.get('notes','')))
        message = "Symptoms logged!"

    with get_db() as db:
        logs = db.execute(
            "SELECT * FROM symptoms WHERE user_id=? ORDER BY log_date DESC LIMIT 14",(uid,)
        ).fetchall()

    return render_template('symptoms.html', logs=logs, message=message,
                           today=datetime.today().strftime('%Y-%m-%d'),
                           user_name=session.get('user_name',''))

if __name__ == '__main__':
    import webbrowser, threading
    threading.Timer(1, lambda: webbrowser.open('http://127.0.0.1:5000/')).start()
    app.run(debug=True, use_reloader=False)
