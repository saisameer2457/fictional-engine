import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test the index route"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to the LeNet-5 Prediction API!" in response.data

def test_predict(client):
    """Test the predict route"""
    data = {
        "features": [0] * (1 * 1 * 28 * 28)
    }
    response = client.post('/predict', json=data)
    assert response.status_code == 200
    assert 'prediction' in response.get_json()
