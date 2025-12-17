import sqlite3
import hashlib
import json
from datetime import datetime

DB_NAME = "heart_guard.db"

def init_db():
    """Initialize the database with necessary tables."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Create Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            age INTEGER,
            sex TEXT,
            weight_kg REAL,
            height_cm REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create Medical Records Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS medical_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            medical_data TEXT, -- JSON string of the medical dict
            prediction_result INTEGER,
            probability REAL,
            risk_level TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, full_name, age, sex, weight, height):
    """Register a new user."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        pwd_hash = hash_password(password)
        
        c.execute('''
            INSERT INTO users (username, password_hash, full_name, age, sex, weight_kg, height_cm)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (username, pwd_hash, full_name, age, sex, weight, height))
        
        conn.commit()
        return True, "User created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def verify_user(username, password):
    """Verify login credentials."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # To access columns by name
    c = conn.cursor()
    
    pwd_hash = hash_password(password)
    
    c.execute('SELECT * FROM users WHERE username = ? AND password_hash = ?', (username, pwd_hash))
    user = c.fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None

def add_medical_record(user_id, medical_data, prediction_result):
    """Save a prediction result to history."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # medical_data is a dict, convert to JSON string
    medical_json = json.dumps(medical_data)
    
    c.execute('''
        INSERT INTO medical_records (user_id, medical_data, prediction_result, probability, risk_level)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        user_id, 
        medical_json, 
        prediction_result['prediction'], 
        prediction_result['probability_1'],
        prediction_result['risk_level']
    ))
    
    conn.commit()
    conn.close()

def get_user_history(user_id):
    """Retrieve all medical records for a user."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM medical_records 
        WHERE user_id = ? 
        ORDER BY timestamp DESC
    ''', (user_id,))
    
    rows = c.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        r = dict(row)
        r['medical_data'] = json.loads(r['medical_data'])
        history.append(r)
        
    return history
