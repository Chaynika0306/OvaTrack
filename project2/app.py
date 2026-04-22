from flask import Flask, render_template, request, session, redirect, url_for, jsonify
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
        CREATE TABLE IF NOT EXISTS otp_store (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            email     TEXT NOT NULL,
            otp       TEXT NOT NULL,
            expires   TEXT NOT NULL,
            used      INTEGER DEFAULT 0
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
def landing():
    # If already logged in, go straight to app
    if session.get('user_id'):
        return redirect(url_for('home'))
    return render_template('landing.html')

@app.route('/app')
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

# ── AI CHATBOT — Smart rule-based responses (no API key needed) ─────────────────
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    import re, random

    user_msg = request.json.get('message', '').strip().lower()
    if not user_msg:
        return jsonify({'reply': 'Please type a message.'})

    uid  = session['user_id']
    dash = session.get('dash', {})

    # ── Build personalised context prefix ────────────────────────────────
    ctx = ""
    if dash.get('type') == 'pcos' and dash.get('risk'):
        r = dash['risk']
        rl = dash.get('risk_level','')
        ctx = f"(Based on your {r}% {rl} PCOS risk) "

    # ── Keyword → response bank ───────────────────────────────────────────
    responses = [
        # Diet / food
        (r'diet|food|eat|khaana|khana|nutrition|glycemic',
         [ctx + "For PCOS, focus on low-glycemic foods — oats, lentils, vegetables, fruits, and lean proteins 🥗 Avoid refined sugar, white bread, and processed snacks. Small frequent meals help stabilise blood sugar and hormones.",
          ctx + "The best PCOS diet includes: fibre-rich foods (vegetables, legumes), lean proteins (eggs, chicken, tofu), healthy fats (nuts, avocado), and complex carbs (brown rice, millets) 🌿 Cut back on sugary drinks and fast food.",
          ctx + "Eating anti-inflammatory foods helps PCOS — think turmeric, leafy greens, berries, and fish. Cutting sugar and processed carbs is one of the most impactful dietary changes you can make 💚"]),

        # Exercise / workout
        (r'exercise|workout|gym|yoga|walk|physical|fit|active|vyayaam',
         [ctx + "For PCOS, aim for 30 minutes of moderate exercise 5 days a week 🏃‍♀️ Walking, yoga, cycling, and swimming are all excellent. Resistance training also helps improve insulin sensitivity.",
          ctx + "Yoga is especially beneficial for PCOS — poses like Butterfly, Cobra, and Child's pose help regulate hormones and reduce stress 🧘‍♀️ Even a 20-minute daily walk makes a meaningful difference.",
          ctx + "Exercise helps lower insulin levels, reduce androgens, and regulate your cycle 💪 Combine cardio with light strength training 3-4 times a week for the best hormonal results."]),

        # Stress / anxiety / mental health
        (r'stress|anxiety|mental|mood|sad|depressed|overwhelm|tension|pareshan',
         [ctx + "Stress raises cortisol, which directly worsens PCOS symptoms 😔 Try 10 minutes of deep breathing or meditation daily. Apps like Headspace can help if you're just starting out.",
          ctx + "High stress disrupts the hormonal balance that regulates your cycle 💆‍♀️ Prioritise 7-8 hours of sleep, reduce screen time before bed, and try journaling to process emotions.",
          ctx + "You're not alone — anxiety and mood swings are very common with PCOS due to hormonal imbalance 💙 Gentle yoga, walks in nature, and talking to someone you trust can make a real difference."]),

        # Sleep
        (r'sleep|nind|neend|rest|tired|thakaan|fatigue|insomnia',
         [ctx + "Sleep is critical for hormone regulation 😴 Aim for 7-9 hours per night. Poor sleep increases cortisol and insulin resistance — both of which worsen PCOS symptoms.",
          ctx + "For better sleep with PCOS, try keeping a consistent bedtime, avoiding screens 1 hour before bed, and keeping your room cool and dark 🌙 Magnesium supplements can also help some people.",
          ctx + "PCOS fatigue is very real — your body is working harder due to hormonal imbalance 💤 Prioritise sleep hygiene, stay hydrated, and consider a gentle 10-minute walk after meals to boost energy."]),

        # Irregular periods / cycle
        (r'period|cycle|irregular|late|miss|menstrual|menses|mahavari|masik',
         [ctx + "Irregular periods are the most common PCOS symptom 🌊 Lifestyle changes — diet, exercise, and stress reduction — can help regulate cycles naturally. Consult a gynecologist if periods are missing for 3+ months.",
          ctx + "Tracking your cycle with OvaTrack can reveal patterns even when cycles seem irregular 📅 Use our Cycle Predictor to get an AI estimate of your next period based on your history.",
          ctx + "Hormonal balance takes time to restore 🌸 Many women see cycle improvements within 3-6 months of consistent lifestyle changes. Inositol supplements are also commonly recommended — ask your doctor."]),

        # Weight / BMI
        (r'weight|bmi|fat|obesity|slim|lose weight|vajan|mota',
         [ctx + "Weight management with PCOS is harder because of insulin resistance — but it's possible ⚖️ Even a 5-10% reduction in body weight can significantly improve hormonal balance and cycle regularity.",
          ctx + "Focus on sustainable changes rather than crash diets 🥗 Reducing sugar, increasing protein and fibre, and exercising regularly is more effective long-term than restrictive dieting for PCOS.",
          ctx + "PCOS makes it harder to lose weight due to elevated insulin levels 💪 A combination of low-glycemic eating + regular movement + stress management works better than dieting alone."]),

        # Acne / skin / hair
        (r'acne|pimple|skin|hair|thinning|facial hair|hirsutism|daag|baal',
         [ctx + "Acne and excess hair growth in PCOS are caused by high androgen levels 😣 Spearmint tea has been shown to reduce androgens naturally. A dermatologist can also recommend targeted treatments.",
          ctx + "For hormonal acne, avoid high-glycemic foods (sugar, white bread) which spike insulin and androgens 💆 Keep skin clean, use non-comedogenic products, and stay hydrated.",
          ctx + "Hair thinning with PCOS is linked to DHT — a type of androgen 💇 Iron and Vitamin D deficiency can worsen it. Get blood tests done and discuss options like minoxidil with your doctor."]),

        # PCOS general / what is
        (r'what is pcos|pcos kya|pcos ke baare|polycystic|ovary|symptoms of',
         [ctx + "PCOS (Polycystic Ovary Syndrome) is a hormonal disorder affecting 1 in 10 women 🌸 It causes irregular periods, excess androgens, and small ovarian cysts. It's manageable with lifestyle changes and medical support.",
          ctx + "PCOS symptoms include irregular cycles, acne, hair thinning, weight gain, fatigue, and mood changes 📋 Not every woman has all symptoms. OvaTrack's AI analyses your specific pattern to estimate your risk.",
          ctx + "PCOS is the most common cause of female infertility — but it's very treatable 💚 Early detection and lifestyle changes make a huge difference. You're already taking the right step by tracking your health!"]),

        # Ovulation / fertility
        (r'ovulation|fertile|fertility|pregnant|baby|conceive|garbh',
         [ctx + "With PCOS, ovulation can be irregular or absent 🌕 Tracking basal body temperature and using OvulationTrack can help identify your fertile window. Consult a doctor if you're trying to conceive.",
          ctx + "Improving insulin sensitivity through diet and exercise often restores ovulation naturally in PCOS 💪 Many women with PCOS conceive naturally after lifestyle changes. Medical options like Clomid are also effective.",
          ctx + "Your fertile window is typically around Day 14 of your cycle 📅 With PCOS, this can shift. OvaTrack's notification system estimates your ovulation window based on your cycle data."]),

        # Vitamin / supplements
        (r'vitamin|supplement|inositol|zinc|magnesium|omega|vitamin d',
         [ctx + "For PCOS, commonly recommended supplements include: Inositol (improves insulin sensitivity), Vitamin D (often deficient), Omega-3 (reduces inflammation), and Zinc (helps with acne and hair loss) 💊 Always consult a doctor before starting.",
          ctx + "Myo-inositol is one of the most researched supplements for PCOS 🌿 It improves insulin sensitivity, reduces androgen levels, and can help regulate cycles. A typical dose is 2-4g daily.",
          ctx + "Vitamin D deficiency is extremely common in PCOS women 🌞 Get your levels tested — many doctors recommend supplementation. It plays a key role in insulin regulation and hormonal balance."]),

        # Doctor / medication / treatment
        (r'doctor|medication|medicine|treatment|gynecologist|metformin|pill|dawai',
         [ctx + "Please consult a gynecologist or endocrinologist for medical treatment of PCOS 👩‍⚕️ Common medical options include Metformin (insulin sensitizer), oral contraceptives (cycle regulation), and anti-androgens.",
          ctx + "OvaTrack gives you a risk assessment to start the conversation with your doctor 📋 Bring your OvaTrack report to your appointment — it shows risk percentage, BMI, cycle data, and symptom history.",
          ctx + "A comprehensive PCOS workup usually includes blood tests (FSH, LH, testosterone, insulin, thyroid) and an ultrasound 🩺 Your OvaTrack data can guide your doctor to the right tests."]),
    ]

    # ── Greeting special case ──────────────────────────────────────────────
    if re.search(r'^(hi|hello|hey|namaste|hii|helo|good morning|good evening)', user_msg):
        notif = get_notification(uid)
        notif_msg = f" {notif['message']}" if notif and notif.get('message') else ""
        greetings = [
            f"Hi! I'm OvaTrack AI 🌸 I'm here to help with anything about PCOS, your cycle, diet, or hormonal health.{notif_msg} What would you like to know?",
            f"Hello! Great to see you 😊 I'm your OvaTrack health assistant. I can answer questions about PCOS, nutrition, cycle tracking, or general women's health.{notif_msg}",
            f"Namaste! 🙏 I'm here to support your health journey. Ask me anything about PCOS, diet, exercise, or your menstrual cycle!{notif_msg}"
        ]
        return jsonify({'reply': random.choice(greetings)})

    # ── Match keywords ─────────────────────────────────────────────────────
    for pattern, replies in responses:
        if re.search(pattern, user_msg):
            return jsonify({'reply': random.choice(replies)})

    # ── Personalised default with context ──────────────────────────────────
    defaults = [
        ctx + "That's a great question! For personalised advice, I recommend discussing this with your gynecologist. In general, managing PCOS involves balanced diet, regular exercise, stress reduction, and good sleep 🌸",
        ctx + "I may not have a specific answer for that, but I'm here to help with PCOS, cycle tracking, diet, hormones, and lifestyle questions 💚 Could you rephrase or ask something more specific?",
        ctx + "For women's health questions I'm fully equipped to help! Try asking about PCOS symptoms, diet tips, exercise recommendations, period irregularities, or stress management 🌿",
    ]
    return jsonify({'reply': random.choice(defaults)})



# ── NOTIFICATIONS API ─────────────────────────────────────────────────────────────
@app.route('/api/notifications')
@login_required
def api_notifications():
    uid   = session['user_id']
    notifs = []

    with get_db() as db:
        cycle = db.execute(
            "SELECT * FROM cycle_log WHERE user_id=? ORDER BY created DESC LIMIT 1", (uid,)
        ).fetchone()

    if cycle and cycle['next_period']:
        today      = datetime.today().date()
        next_p     = datetime.strptime(cycle['next_period'], '%Y-%m-%d').date()
        last_p     = datetime.strptime(cycle['last_period'], '%Y-%m-%d').date()
        cycle_len  = int(cycle['cycle_length'])
        delta      = (next_p - today).days

        # ── Period reminder
        if delta < 0:
            notifs.append({
                'type': 'period', 'level': 'danger',
                'icon': '🔴', 'title': 'Period Overdue',
                'body': f"Your period was expected {abs(delta)} day(s) ago ({cycle['next_period']}). If significantly late, consult a doctor.",
                'days': abs(delta)
            })
        elif delta == 0:
            notifs.append({
                'type': 'period', 'level': 'warning',
                'icon': '🩸', 'title': 'Period Expected Today',
                'body': "Your period is expected today. Stock up on supplies and take it easy!",
                'days': 0
            })
        elif delta <= 3:
            notifs.append({
                'type': 'period', 'level': 'warning',
                'icon': '⏰', 'title': f'Period in {delta} Day(s)',
                'body': f"Your next period is due on {cycle['next_period']}. Prepare in advance.",
                'days': delta
            })
        elif delta <= 7:
            notifs.append({
                'type': 'period', 'level': 'info',
                'icon': '📅', 'title': f'Period in {delta} Days',
                'body': f"Upcoming period on {cycle['next_period']}. Stay hydrated and manage stress.",
                'days': delta
            })

        # ── Ovulation alert (approx. day 14 from last period = cycle_len - 14 days before next)
        ovulation_date = last_p + timedelta(days=14)
        ov_delta = (ovulation_date - today).days
        if -1 <= ov_delta <= 3:
            notifs.append({
                'type': 'ovulation', 'level': 'success',
                'icon': '🌕', 'title': 'Ovulation Window',
                'body': f"You are {'in' if ov_delta <= 0 else 'approaching'} your estimated ovulation window ({ovulation_date.strftime('%b %d')}). This is your most fertile period.",
                'days': max(0, ov_delta)
            })
        elif 4 <= ov_delta <= 7:
            notifs.append({
                'type': 'ovulation', 'level': 'info',
                'icon': '🌖', 'title': f'Ovulation in ~{ov_delta} Days',
                'body': f"Estimated ovulation around {ovulation_date.strftime('%b %d')}. Your fertile window is approaching.",
                'days': ov_delta
            })

        # ── Luteal phase tip (7 days before period)
        if 5 <= delta <= 10:
            notifs.append({
                'type': 'tip', 'level': 'tip',
                'icon': '💆', 'title': 'Luteal Phase — Self Care Week',
                'body': "You're in the luteal phase. Common PMS symptoms may appear. Prioritize rest, magnesium-rich foods, and gentle exercise.",
                'days': delta
            })

    # ── Health tip if no cycle data
    if not notifs:
        notifs.append({
            'type': 'tip', 'level': 'info',
            'icon': '💡', 'title': 'Track Your Cycle',
            'body': "Enter your last period date in the PCOS or Cycle form to get personalized period reminders and ovulation alerts.",
            'days': None
        })

    return jsonify(notifs)




# ── FORGOT PASSWORD ────────────────────────────────────────────────────────────
@app.route('/forgot-password', methods=['GET','POST'])
def forgot_password():
    import smtplib, random, string
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    msg_sent = None
    error    = None

    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        with get_db() as db:
            user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

        if not user:
            error = "No account found with that email address."
        else:
            # Generate 6-digit OTP
            otp = ''.join(random.choices(string.digits, k=6))
            expires = (datetime.now() + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')

            with get_db() as db:
                db.execute("DELETE FROM otp_store WHERE email=?", (email,))
                db.execute("INSERT INTO otp_store (email,otp,expires) VALUES (?,?,?)",
                           (email, otp, expires))

            # Send email via Gmail
            try:
                sender    = "ovatrack9903@gmail.com"
                app_pass  = "xvvy ithm wrdx yxck"   # Gmail App Password
                recipient = email

                mail_msg = MIMEMultipart("alternative")
                mail_msg['Subject'] = "OvaTrack — Password Reset OTP"
                mail_msg['From']    = sender
                mail_msg['To']      = recipient

                html_body = f"""
                <div style="font-family:Arial,sans-serif;max-width:480px;margin:0 auto;padding:32px;
                             background:#fdf7f2;border-radius:16px;border:1px solid #e8ddef">
                  <div style="text-align:center;margin-bottom:24px">
                    <div style="font-size:36px">🌸</div>
                    <h2 style="font-family:Georgia,serif;color:#3d1f4e;margin:8px 0">OvaTrack</h2>
                  </div>
                  <h3 style="color:#2a1a30;margin-bottom:8px">Password Reset Request</h3>
                  <p style="color:#7a6882;line-height:1.6">
                    We received a request to reset your OvaTrack password.
                    Use the OTP below — it expires in <strong>10 minutes</strong>.
                  </p>
                  <div style="background:#fff;border:2px dashed #e8537a;border-radius:12px;
                               padding:24px;text-align:center;margin:24px 0">
                    <div style="font-size:36px;font-weight:700;letter-spacing:8px;color:#3d1f4e">{otp}</div>
                    <div style="color:#7a6882;font-size:12px;margin-top:6px">One-Time Password</div>
                  </div>
                  <p style="color:#9ca3af;font-size:12px">
                    If you didn't request this, please ignore this email.
                    Your account is safe.
                  </p>
                  <hr style="border:none;border-top:1px solid #e8ddef;margin:20px 0">
                  <p style="color:#c4b8cc;font-size:11px;text-align:center">
                    OvaTrack — AI Women's Health Platform
                  </p>
                </div>"""

                mail_msg.attach(MIMEText(html_body, 'html'))

                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                    server.login(sender, app_pass)
                    server.sendmail(sender, recipient, mail_msg.as_string())

                session['reset_email'] = email
                return redirect(url_for('verify_otp'))

            except Exception as e:
                error = f"Could not send email. Please try again. ({str(e)[:60]})"

    return render_template('forgot_password.html', error=error, msg_sent=msg_sent)


@app.route('/verify-otp', methods=['GET','POST'])
def verify_otp():
    email = session.get('reset_email','')
    if not email:
        return redirect(url_for('forgot_password'))

    error = None

    if request.method == 'POST':
        otp_input = request.form.get('otp','').strip()
        new_pw    = request.form.get('new_password','')
        confirm   = request.form.get('confirm_password','')

        if len(new_pw) < 6:
            error = "Password must be at least 6 characters."
        elif new_pw != confirm:
            error = "Passwords do not match."
        else:
            with get_db() as db:
                record = db.execute(
                    "SELECT * FROM otp_store WHERE email=? AND otp=? AND used=0 ORDER BY id DESC LIMIT 1",
                    (email, otp_input)
                ).fetchone()

            if not record:
                error = "Invalid OTP. Please check and try again."
            elif datetime.strptime(record['expires'], '%Y-%m-%d %H:%M:%S') < datetime.now():
                error = "OTP has expired. Please request a new one."
            else:
                with get_db() as db:
                    db.execute("UPDATE users SET password=? WHERE email=?",
                               (hash_pw(new_pw), email))
                    db.execute("UPDATE otp_store SET used=1 WHERE email=?", (email,))

                session.pop('reset_email', None)
                return redirect(url_for('login', reset=1))

    return render_template('verify_otp.html', email=email, error=error)

# ── LEARN ABOUT PCOS ─────────────────────────────────────────────────────────────
@app.route('/learn-pcos')
def learn_pcos():
    logged_in = bool(session.get('user_id'))
    return render_template('learn_pcos.html',
                           user_name=session.get('user_name',''),
                           logged_in=logged_in)

if __name__ == '__main__':
    import webbrowser, threading
    threading.Timer(1, lambda: webbrowser.open('http://127.0.0.1:5000/')).start()
    app.run(debug=True, use_reloader=False)
