import requests, os
from typing import List, Dict, Any
from datetime import datetime
import logging
import hashlib
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

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
        # Initialize Presidio engines
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler('supabase.log')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

    def anonymize_text(self, text: str) -> str:
        """Anonymize PII (names) in the given text"""
        try:
            # Define entities to look for (only PERSON for names)
            pii_entities = ["PERSON"]
            
            # Analyze text for PII
            analyzer_results = self.analyzer.analyze(
                text=text,
                entities=pii_entities,
                language='en'  # Default to English
            )
            
            # Anonymize the detected PII
            anonymized_results = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators={
                    "PERSON": OperatorConfig("replace", {"new_value": "<NAME>"})
                }
            )
            
            return anonymized_results.text
            
        except Exception as e:
            self.logger.error(f"Failed to anonymize text: {str(e)}")
            return text  # Return original text if anonymization fails
    
    def log_chat(self, mobile: str, user: str, response: str) -> bool:
        """Log a chat message to Supabase"""
        try:
            # Hash the mobile number
            hashed_mobile = hashlib.md5(mobile.encode()).hexdigest()
            
            # Anonymize the user message
            anon_user_message = self.anonymize_text(user)
            
            json_data = {
                'mobile': hashed_mobile,
                'user': anon_user_message,
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
# For Hinglish and Marathi detection

from typing import List, Optional, Tuple
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import json

class MultilingualNameDetector:
    def __init__(self):
        # Initialize either OpenAI client or local LLM
        self.setup_llm()
        
    def setup_llm(self):
        """Setup LLM - either API based or local"""
        try:
            # Option 1: OpenAI
            from openai import OpenAI
            self.llm_client = OpenAI()
            self.llm_type = "openai"
            
            # Option 2: Local LLM (e.g., using ctransformers)
            # from ctransformers import AutoModelForCausalLM
            # self.llm_client = AutoModelForCausalLM.from_pretrained(
            #     "TheBloke/Llama-2-7B-Chat-GGUF",
            #     model_file="llama-2-7b-chat.Q4_K_M.gguf"
            # )
            # self.llm_type = "local"
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {str(e)}")
            self.llm_client = None
    
    def detect_names_llm(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Use LLM to detect names in multilingual text
        Returns list of (name, start_pos, end_pos)
        """
        try:
            if not self.llm_client:
                return []
                
            # Prompt engineering for name detection
            prompt = f"""
            Analyze this text and identify any person names (including Indian names).
            Text: "{text}"
            
            Return only a JSON array of objects with format:
            [{{"name": "detected_name", "start": start_position, "end": end_position}}]
            
            If no names found, return empty array [].
            """
            
            if self.llm_type == "openai":
                response = self.llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a precise name detection tool."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                result = json.loads(response.choices[0].message.content)
                
            else:  # local LLM
                response = self.llm_client.generate(prompt, max_tokens=200)
                result = json.loads(response)
                
            return [(item["name"], item["start"], item["end"]) for item in result]
            
        except Exception as e:
            self.logger.error(f"LLM name detection failed: {str(e)}")
            return []

class SupabaseManager:
    def __init__(self, ...):
        # ... existing init code ...
        self.name_detector = MultilingualNameDetector()
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize PII (names) using both Presidio and LLM"""
        try:
            # 1. First try Presidio for English names
            analyzer_results = self.analyzer.analyze(
                text=text,
                entities=["PERSON"],
                language='en'
            )
            
            # 2. Then use LLM for additional name detection
            llm_detected_names = self.name_detector.detect_names_llm(text)
            
            # 3. Combine results (you'll need to convert LLM results to Presidio format)
            # This is a simplified version - you might need more sophisticated merging
            for name, start, end in llm_detected_names:
                analyzer_results.append(
                    # Create Presidio RecognizerResult object
                    RecognizerResult(
                        entity_type="PERSON",
                        start=start,
                        end=end,
                        score=0.85  # Confidence score
                    )
                )
            
            # 4. Anonymize using combined results
            anonymized_results = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators={
                    "PERSON": OperatorConfig("replace", {"new_value": "<NAME>"})
                }
            )
            
            return anonymized_results.text
            
        except Exception as e:
            self.logger.error(f"Failed to anonymize text: {str(e)}")
            return text


'''