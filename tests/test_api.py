import pytest
from fastapi.testclient import TestClient
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from ..sukoon_api import app # to access relative directory
import logging
from datetime import datetime
import json

from sukoon_api import app
# from api import app

# Set up logging
def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/test_api_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Create a test client
client = TestClient(app)

def log_test_result(test_name: str, request_data: dict = None, response_data: dict = None):
    """Helper function to log test results"""
    log_data = {
        "test_name": test_name,
        "timestamp": datetime.now().isoformat(),
    }
    if request_data:
        log_data["request"] = request_data
    if response_data:
        log_data["response"] = response_data
    
    logger.info(json.dumps(log_data, indent=2))

# @pytest.mark.api
# def test_root_endpoint():
#     """Test the root endpoint (/)"""
#     response = client.get("/")
    
#     log_test_result(
#         "test_root_endpoint",
#         response_data=response.json()
#     )
    
#     assert response.status_code == 200
#     assert "message" in response.json()
#     assert "Welcome to the Sukoon API" in response.json()["message"]

@pytest.mark.api
def test_query_endpoint_success():
    """Test the query endpoint with valid input"""
    test_input = "Hello, how are you?"
    test_mobile = "1234567890"  # Added test mobile number
    
    response = client.post(
        "/query",
        json={"input": test_input, "mobile": test_mobile}  # Updated request format
    )
    
    log_test_result(
        "test_query_endpoint_success",
        request_data={"input": test_input, "mobile": test_mobile},
        response_data=response.json()
    )
    
    assert response.status_code == 200
    assert "output" in response.json()
    assert isinstance(response.json()["output"], str)
    assert len(response.json()["output"]) > 0

@pytest.mark.api
def test_query_endpoint_empty_input():
    """Test the query endpoint with empty input"""
    response = client.post(
        "/query",
        json={"input": "", "mobile": "1234567890"}  # Added required mobile field
    )
    
    log_test_result(
        "test_query_endpoint_empty_input",
        request_data={"input": "", "mobile": "1234567890"},
        response_data=response.json()
    )
    
    assert response.status_code == 200
    assert "output" in response.json()

@pytest.mark.api
def test_query_endpoint_invalid_mobile():  # Added test for invalid mobile
    """Test the query endpoint with invalid mobile number"""
    response = client.post(
        "/query",
        json={"input": "Hello", "mobile": "123"}  # Invalid mobile number
    )
    
    log_test_result(
        "test_query_endpoint_invalid_mobile",
        request_data={"input": "Hello", "mobile": "123"},
        response_data=response.json()
    )
    
    assert response.status_code == 422  # Validation error

@pytest.mark.api
def test_query_endpoint_invalid_request():
    """Test the query endpoint with invalid request format"""
    response = client.post(
        "/query",
        json={"wrong_field": "Hello"}  # Missing both required fields
    )
    
    log_test_result(
        "test_query_endpoint_invalid_request",
        request_data={"wrong_field": "Hello"},
        response_data=response.json()
    )
    
    assert response.status_code == 422

@pytest.mark.api
def test_fetch_history_endpoint():  # Added test for fetch_convo endpoint
    """Test the fetch_convo endpoint"""
    response = client.get("/fetch_convo?mobile=1234567890")
    
    log_test_result(
        "test_fetch_history_endpoint",
        request_data={"mobile": "1234567890"},
        response_data=response.json()
    )
    
    assert response.status_code == 200
    assert "messages" in response.json()
    assert isinstance(response.json()["messages"], list)

if __name__ == "__main__":
    pytest.main(["-v", __file__])