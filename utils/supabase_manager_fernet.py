import requests
from typing import List, Dict, Any
from datetime import datetime
import logging
import hashlib
from cryptography.fernet import Fernet
import base64
import os

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_CLIENT_ANON_KEY")

class SupabaseManager:
    def __init__(self):
        self.url = supabase_url
        self.headers = {
            'apikey': supabase_key,
            'Authorization': f'Bearer {supabase_key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal'
        }
        # Initialize Fernet with a consistent key
        self.fernet_key = os.getenv('FERNET_KEY') # or Fernet.generate_key() if none exists. # In production, this should be stored securely (e.g., environment variables)
        self.fernet = Fernet(self.fernet_key)
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler('supabase.log')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

    def encrypt_mobile(self, mobile: str) -> str:
        """Encrypt mobile number and return as base64 string"""
        encrypted = self.fernet.encrypt(mobile.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_mobile(self, encrypted_mobile: str) -> str:
        """Decrypt mobile number from base64 string"""
        encrypted = base64.urlsafe_b64decode(encrypted_mobile.encode())
        return self.fernet.decrypt(encrypted).decode()
    
    def log_chat(self, mobile: str, user: str, response: str) -> bool:
        """Log a chat message to Supabase"""
        try:
            encrypted_mobile = self.encrypt_mobile(mobile)
            
            # use presidio hre for masking names and other PII
            
            json_data = {
                'mobile': encrypted_mobile,
                'user': user,
                'response': response
            }
            
            response = requests.post(self.url, headers=self.headers, json=json_data)
            response.raise_for_status()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log chat: {str(e)}")
            return False

    def get_chat_history(self, mobile: str) -> List[Dict[str, Any]]:
        """
        Get chat history for a user from Supabase
        
        Args:
            mobile: User's mobile number to fetch chat history
            
        Returns:
            List of dictionaries containing user messages and responses
        """
        try:
            encrypted_mobile = self.encrypt_mobile(mobile)
            # Properly encode the mobile number for URL
            encoded_mobile = requests.utils.quote(encrypted_mobile)
            
            # Build the query with specific columns and conditions
            query_url = (
                f"{self.url}"
                f"?select=user,response"
                f"&mobile=eq.{encoded_mobile}"
                f"&order=created_at.asc"
            )
            
            self.logger.debug(f"Fetching chat history for mobile: {encrypted_mobile}")
            
            response = requests.get(
                query_url,
                headers={
                    **self.headers,
                    'Prefer': 'return=representation'  # Ensures Supabase returns the data
                }
            )
            
            response.raise_for_status()  # Raise exception for non-200 status codes
            
            data = response.json()
            self.logger.debug(f"Retrieved {len(data)} chat records")
            
            # Filter out any records where either user or response is None/empty
            messages = [
                {
                    "user": msg["user"],
                    "response": msg["response"]
                }
                for msg in data
                if msg.get("user") and msg.get("response")  # Only include complete conversations
            ]
            
            return messages
            
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP Error in get_chat_history: {str(e)}")
            if response.status_code == 409:
                self.logger.error("Conflict error - possible duplicate or constraint violation")
            return []
            
        except Exception as e:
            self.logger.error(f"Error in get_chat_history: {str(e)}")
            return []
                
'''
Key Points to Remember:

Key Management:

 # Bad - don't do this:
KEY = Fernet.generate_key()  # Hardcoded in code

# Better - use environment variables:
import os
KEY = os.environ.get('ENCRYPTION_KEY')

# Even Better - use a key management service:
from aws_kms_provider import KMSProvider  # Example
key_provider = KMSProvider()
KEY = key_provider.get_key()

Error Handling:

 from cryptography.fernet import InvalidToken

def safe_decrypt_mobile(key: bytes, encrypted_mobile: bytes) -> str:
    try:
        fernet = Fernet(key)
        decrypted_mobile = fernet.decrypt(encrypted_mobile)
        return decrypted_mobile.decode()
    except InvalidToken:
        raise ValueError("Invalid key or corrupted encrypted data")
    except Exception as e:
        raise Exception(f"Decryption failed: {str(e)}")

Key Rotation:

 def rotate_encryption(old_key: bytes, new_key: bytes, encrypted_data: bytes) -> bytes:
    """Rotate encrypted data to use a new key"""
    old_fernet = Fernet(old_key)
    new_fernet = Fernet(new_key)
    
    # Decrypt with old key
    temp_decrypted = old_fernet.decrypt(encrypted_data)
    
    # Encrypt with new key
    return new_fernet.encrypt(temp_decrypted)

'''