"""
Dirty Loader Module
Extracts data from malformed/broken JSON files using regex patterns
"""

import re
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DirtyLoader:
    """
    Extracts text content from files without parsing JSON.
    Survives broken JSON, missing brackets, and huge file sizes.
    """
    
    @staticmethod
    def extract_context_text(file_path: str) -> str:
        """
        Extracts text content directly from a file without parsing it as JSON.
        
        Args:
            file_path: Path to the (possibly broken) JSON file
            
        Returns:
            Extracted text content as a single string
        """
        logger.info(f"[DIRTY LOADER] Extracting text from {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_data = f.read()
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return ""
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""
        
        logger.info(f"File loaded: {len(raw_data)} characters")
        
        # STRATEGY 1: Target specific context keys
        # Look for common context field names
        pattern = r'"(?:text|content|snippet|page_content|context|chunk|passage|document)"\s*:\s*"((?:[^"\\]|\\.)*)"'
        
        matches = re.finditer(pattern, raw_data, re.IGNORECASE | re.DOTALL)
        
        extracted_chunks = []
        for match in matches:
            # Unescape the content
            clean_content = match.group(1).replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            if clean_content.strip():  # Skip empty strings
                extracted_chunks.append(clean_content)
        
        if extracted_chunks:
            logger.info(f"[SUCCESS] Extracted {len(extracted_chunks)} text chunks via targeted regex")
            combined = "\n\n".join(extracted_chunks)
            logger.info(f"Total extracted text: {len(combined)} characters")
            return combined
        
        # STRATEGY 2: Fallback - Extract all substantial strings
        logger.warning("[FALLBACK] Targeted extraction failed. Dumping all long strings...")
        generic_string_pattern = r'"((?:[^"\\]|\\.)*)"'
        matches = re.finditer(generic_string_pattern, raw_data)
        
        texts = []
        for match in matches:
            s = match.group(1)
            # Filter: keep only strings that look like sentences (>50 chars, has spaces)
            if len(s) > 50 and " " in s and not s.startswith("http"):
                cleaned = s.replace('\\"', '"').replace('\\n', '\n')
                texts.append(cleaned)
        
        if texts:
            logger.info(f"[FALLBACK SUCCESS] Extracted {len(texts)} text chunks via generic pattern")
            combined = "\n\n".join(texts)
            logger.info(f"Total extracted text: {len(combined)} characters")
            return combined
        
        logger.error("Failed to extract any meaningful text from file")
        return ""
    
    @staticmethod
    def extract_conversation_messages(file_path: str) -> List[Dict]:
        """
        Extracts messages from conversation file using pattern matching.
        More robust than JSON parsing.
        
        Args:
            file_path: Path to conversation JSON file
            
        Returns:
            List of message dictionaries
        """
        logger.info(f"[DIRTY LOADER] Extracting messages from {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_data = f.read()
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
        
        logger.info(f"File loaded: {len(raw_data)} characters")
        
        # Try normal JSON parsing first
        try:
            data = json.loads(raw_data)
            logger.info("[SUCCESS] Parsed as valid JSON")
            
            if isinstance(data, dict) and 'messages' in data:
                logger.info(f"Found {len(data['messages'])} messages in 'messages' key")
                return data['messages']
            elif isinstance(data, dict) and 'conversation_turns' in data:
                logger.info(f"Found {len(data['conversation_turns'])} messages in 'conversation_turns' key")
                return data['conversation_turns']
            elif isinstance(data, list):
                logger.info(f"Data is a list with {len(data)} messages")
                return data
            else:
                logger.warning(f"JSON structure unrecognized. Keys: {list(data.keys())}")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            logger.info("Attempting regex extraction...")
        
        # FALLBACK STRATEGY 1: Extract message objects with flexible patterns
        messages = []
        
        # Pattern 1: Standard message objects with various field names
        patterns = [
            # Pattern: "message": "text", "sender"/"role": "value"
            r'\{[^}]*?"message"\s*:\s*"((?:[^"\\]|\\.)*)"[^}]*?"(?:sender|role|sender_id)"\s*:\s*"?([^",}]+)"?[^}]*?\}',
            
            # Pattern: "content": "text", "sender"/"role": "value"
            r'\{[^}]*?"content"\s*:\s*"((?:[^"\\]|\\.)*)"[^}]*?"(?:sender|role|sender_id)"\s*:\s*"?([^",}]+)"?[^}]*?\}',
            
            # Pattern: "text": "text", "sender"/"role": "value"
            r'\{[^}]*?"text"\s*:\s*"((?:[^"\\]|\\.)*)"[^}]*?"(?:sender|role|sender_id)"\s*:\s*"?([^",}]+)"?[^}]*?\}',
        ]
        
        for pattern in patterns:
            logger.info(f"Trying pattern: {pattern[:50]}...")
            matches = list(re.finditer(pattern, raw_data, re.DOTALL | re.IGNORECASE))
            
            if matches:
                logger.info(f"Found {len(matches)} matches with this pattern")
                
                for match in matches:
                    message_text = match.group(1).replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                    sender = match.group(2).strip()
                    
                    # Try to extract ID if present
                    msg_obj = match.group(0)
                    id_match = re.search(r'"(?:id|message_id|turn)"\s*:\s*"?(\w+)"?', msg_obj)
                    msg_id = id_match.group(1) if id_match else None
                    
                    messages.append({
                        'message': message_text,
                        'sender': sender,
                        'id': msg_id
                    })
                
                if messages:
                    logger.info(f"[FALLBACK SUCCESS] Extracted {len(messages)} messages")
                    return messages
        
        # FALLBACK STRATEGY 2: Just find all "message"/"content" fields and guess roles
        logger.warning("Structured extraction failed. Trying generic text extraction...")
        
        text_pattern = r'"(?:message|content|text)"\s*:\s*"((?:[^"\\]|\\.)*)"'
        text_matches = re.finditer(text_pattern, raw_data, re.IGNORECASE)
        
        alternate_role = True
        for match in text_matches:
            text = match.group(1).replace('\\"', '"').replace('\\n', '\n')
            
            # Skip very short texts (likely metadata)
            if len(text) < 10:
                continue
            
            # Alternate between user and AI
            role = "user" if alternate_role else "ai"
            alternate_role = not alternate_role
            
            messages.append({
                'message': text,
                'sender': role,
                'id': None
            })
        
        if messages:
            logger.info(f"[GENERIC EXTRACTION] Found {len(messages)} text fields")
            return messages
        
        logger.error("Failed to extract any messages from conversation file")
        return []
    
    @staticmethod
    def safe_json_load(file_path: str) -> Optional[Dict]:
        """
        Attempts to load JSON with multiple fallback strategies.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON dict or None if all methods fail
        """
        logger.info(f"[SAFE LOAD] Attempting to load {file_path}...")
        
        # Strategy 1: Normal JSON parsing
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info("[SUCCESS] Loaded as valid JSON")
            
            # Debug: Print structure info
            if isinstance(data, dict):
                logger.info(f"   JSON keys: {list(data.keys())}")
                # Check for messages in various keys
                for key in ['messages', 'conversation_turns', 'data', 'conversation']:
                    if key in data:
                        value = data[key]
                        if isinstance(value, list):
                            logger.info(f"   Found '{key}' with {len(value)} items")
                        else:
                            logger.info(f"   Found '{key}' but it's not a list: {type(value)}")
            elif isinstance(data, list):
                logger.info(f"   JSON is a list with {len(data)} items")
            
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode failed at line {e.lineno}, col {e.colno}: {e.msg}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading JSON: {e}")
            return None
        
        # Strategy 2: Try to fix common JSON issues
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_data = f.read()
            
            logger.info("[ATTEMPTING] Fixing common JSON issues...")
            
            # Remove trailing commas
            fixed_data = re.sub(r',\s*}', '}', raw_data)
            fixed_data = re.sub(r',\s*]', ']', fixed_data)
            
            # Try parsing again
            data = json.loads(fixed_data)
            logger.info("[SUCCESS] Loaded after fixing trailing commas")
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"Still cannot parse after basic fixes: {e}")
        except Exception as e:
            logger.error(f"Error during JSON fixing: {e}")
        
        # Strategy 3: Extract as much structure as possible
        logger.warning("[GIVING UP] Cannot load as JSON, will use regex extraction")
        return None


class RobustDataLoader:
    """
    High-level loader that combines DirtyLoader with intelligent fallbacks.
    Use this as the main interface for loading evaluation data.
    """
    
    def __init__(self):
        self.dirty_loader = DirtyLoader()
    
    def load_context(self, file_path: str) -> str:
        """
        Load context data (robust against malformed JSON).
        
        Args:
            file_path: Path to context file
            
        Returns:
            Combined context text
        """
        logger.info("=" * 60)
        logger.info("LOADING CONTEXT DATA (ROBUST MODE)")
        logger.info("=" * 60)
        
        # For context files, always use dirty extraction
        # Context doesn't need structure - just the text content
        context_text = self.dirty_loader.extract_context_text(file_path)
        
        if not context_text:
            logger.error("No context extracted from file")
            return "No context available"
        
        logger.info(f" Context loaded successfully: {len(context_text)} characters")
        return context_text
    
    def load_conversation(self, file_path: str) -> Dict:
        """
        Load conversation data (tries JSON first, falls back to regex).
        
        Args:
            file_path: Path to conversation file
            
        Returns:
            Conversation dictionary with messages
        """
        logger.info("=" * 60)
        logger.info("LOADING CONVERSATION DATA (ROBUST MODE)")
        logger.info("=" * 60)
        
        # Try safe JSON load first
        data = self.dirty_loader.safe_json_load(file_path)
        
        if data and isinstance(data, dict):
            # Check if it actually has messages
            if 'messages' in data and data['messages']:
                logger.info(f"Conversation loaded as valid JSON with {len(data['messages'])} messages")
                return data
            elif 'conversation_turns' in data and data['conversation_turns']:
                logger.info(f"Conversation loaded as valid JSON with {len(data['conversation_turns'])} turns")
                return {"messages": data['conversation_turns']}
            else:
                logger.warning("JSON loaded but no messages found in standard keys")
                # Check if data itself is a list
                if isinstance(data, list):
                    logger.info(f"JSON is a list with {len(data)} items")
                    return {"messages": data}
        
        # If data is a list directly
        if data and isinstance(data, list):
            logger.info(f"Conversation loaded as JSON list with {len(data)} messages")
            return {"messages": data}
        
        # Fallback to regex extraction
        logger.warning("JSON parsing incomplete, using regex extraction for messages")
        messages = self.dirty_loader.extract_conversation_messages(file_path)
        
        if not messages:
            logger.error("FAILED to extract any messages from conversation")
            logger.error("Please check the conversation file format")
            raise ValueError("No messages could be extracted from conversation file")
        
        logger.info(f"Extracted {len(messages)} messages via fallback method")
        return {"messages": messages}
    
    def validate_loaded_data(self, conversation: Dict, context: str) -> tuple[bool, str]:
        """
        Validate that loaded data is usable for evaluation.
        
        Args:
            conversation: Loaded conversation data
            context: Loaded context text
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check conversation
        if not conversation or not isinstance(conversation, dict):
            return False, "Invalid conversation structure"
        
        messages = conversation.get('messages', [])
        if not messages:
            return False, "No messages found in conversation"
        
        # Check context
        if not context or len(context) < 10:
            return False, "Context is empty or too short"
        
        logger.info("Data validation passed")
        return True, "Valid"