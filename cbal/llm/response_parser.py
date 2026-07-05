import json
import re

class ResponseParser:
    @staticmethod
    def parse(response_text):
        """
        Extracts JSON object from text, handling common LLM formatting errors.
        """
        # Default fallback
        fallback = {"action": "KEEP", "confidence": 0.0, "reason": "Parse Error"}
        
        if not response_text:
            return fallback

        try:
            # 1. Try direct JSON load
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # 2. Regex search for JSON block
        try:
            # Matches content between { and } including newlines
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Clean potential trailing commas or markdown
                return json.loads(json_str)
        except:
            pass

        return fallback