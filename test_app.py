import pytest
import sqlite3
import pandas as pd
from app import eval_metrics

def test_eval_metrics():
    actual = [3, 4, 5]
    pred = [3, 4, 5]
    rmse, mae, r2 = eval_metrics(actual, pred)
    assert rmse == 0
    assert mae == 0
    assert r2 == 1

def test_database():
    conn = sqlite3.connect('wine_quality.db')
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    assert c.fetchone() is not None

    # Insert a dummy record
    c.execute("INSERT INTO predictions (fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality, prediction) VALUES (0,0,0,0,0,0,0,0,0,0,0,0,0)")
    conn.commit()

    c.execute("SELECT * FROM predictions WHERE fixed_acidity=0")
    assert c.fetchone() is not None
    conn.close()
