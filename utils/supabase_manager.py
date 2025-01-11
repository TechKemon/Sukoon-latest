import requests, os
from typing import List, Dict, Any
from datetime import datetime
import logging

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
            json_data = {
                'mobile': mobile,
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
            # Properly encode the mobile number for URL
            encoded_mobile = requests.utils.quote(mobile)
            
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
    
    # def get_chat_history(self, mobile: str) -> List[Dict[str, Any]]:
    #     """Get chat history for a user from Supabase"""
    #     try:
    #         # Build the URL with query parameters
    #         query_url = f"{self.url}?select=user,response&mobile=eq.{mobile}&order=created_at.asc"
            
    #         # Log the request details for debugging
    #         print(f"\nFetching chat history:")
            
    #         response = requests.get(
    #             query_url,
    #             headers=self.headers
    #         )
            
    #         # Log response details
    #         print(f"\nResponse status: {response.status_code}")
    #         print(f"Response headers: {response.headers}")
            
    #         # Check if response is successful
    #         if response.status_code == 200:
    #             data = response.json()
    #             print(f"\nRaw response data: {data}")
                
    #             messages = [
    #                 {
    #                     "user": msg["user"],
    #                     "response": msg["response"]
    #                 }
    #                 for msg in data
    #             ]
                
    #             print(f"\nFormatted messages: {messages}")
    #             self.logger.info(f"Successfully retrieved {len(messages)} messages for {mobile}")
    #             return messages
                
    #         else:
    #             print(f"\nError response: {response.text}")
    #             self.logger.error(f"Failed to get chat history. Status code: {response.status_code}")
    #             self.logger.error(f"Response: {response.text}")
    #             return []
                
    #     except requests.exceptions.RequestException as e:
    #         print(f"\nRequest error: {str(e)}")
    #         self.logger.error(f"Request failed: {str(e)}")
    #         return []
            
    #     except Exception as e:
    #         print(f"\nUnexpected error: {str(e)}")
    #         self.logger.error(f"Unexpected error: {str(e)}")
    #         return []
                
       