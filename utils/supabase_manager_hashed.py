import requests, os
from typing import List, Dict, Any
from datetime import datetime
import logging
import hashlib

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
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler('supabase.log')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

    def log_chat(self, mobile: str, user: str, response: str) -> bool:
        """Log a chat message to Supabase"""
        try:
            # Hash the mobile number
            hashed_mobile = hashlib.md5(mobile.encode()).hexdigest()
            
            # use presidio hre for masking names and other PII
            
            json_data = {
                'mobile': hashed_mobile,
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
            # Hash the mobile number for querying
            hashed_mobile = hashlib.md5(mobile.encode()).hexdigest()
            # Properly encode the mobile number for URL
            encoded_mobile = requests.utils.quote(hashed_mobile)
            
            # Build the query with specific columns and conditions
            query_url = (
                f"{self.url}"
                f"?select=user,response"
                f"&mobile=eq.{encoded_mobile}"
                f"&order=created_at.asc"
            )
            
            self.logger.debug(f"Fetching chat history for mobile: {mobile}")
            
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
# MD5 will always generate the same hash for the same input. uses 128 bits (32 hex characters)
For more security, use SHA-256 from same hashlib library

mobile1 = "1234567890"
mobile2 = "9876543210"

hash1 = hashlib.md5(mobile1.encode()).hexdigest()  # e807f1fcf82d132f9bb018ca6738a19f
hash2 = hashlib.md5(mobile2.encode()).hexdigest()  # 0fea51d5a21a1a4547342c81239a7c11

# for salting 
def salt_and_hash(mobile):
    salt = os.urandom(32)  # Generate a random salt
    key = hashlib.pbkdf2_hmac(
        'sha256', 
        str(mobile).encode('utf-8'), 
        salt, 
        100000  # Number of iterations
    )
    return salt.hex() + key.hex()  # Store both salt and key

'''