import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class GemmaAgent:
    def __init__(self, config):
        model_id = config.get('llm_repo', "google/gemma-3-4b-it")
        print(f"🤖 Loading LLM: {model_id}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True
        )
        self.model.eval() 
        self.config = config
        self._parse_failures = 0
        self._total_calls = 0

    def predict(self, prompt_text):
        self._total_calls += 1

        messages = [
            {"role": "user", "content": prompt_text}
        ]

        encoded = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True
        )
        if hasattr(encoded, 'input_ids'):
            input_ids = encoded.input_ids.to("cuda")
        elif isinstance(encoded, dict):
            input_ids = encoded["input_ids"].to("cuda")
        else:
            input_ids = encoded.to("cuda")

        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.get('max_new_tokens', 512),
                do_sample=False, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        input_len = input_ids.shape[1]
        raw_response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        return self._parse_json_robust(raw_response)

    def _parse_json_robust(self, text):
        clean_text = text.strip()
        
        if "```" in clean_text:
            match = re.search(r"```(?:\w+)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
            if match:
                clean_text = match.group(1)

        def try_parse_and_repair(candidate_str):
            try:
                return json.loads(candidate_str)
            except json.JSONDecodeError:
                try:
                    repaired = re.sub(r'(?<=:\s")(.+?)(?=",\n|"\s*\})', lambda m: m.group(1).replace('"', "'"), candidate_str, flags=re.DOTALL)
                    return json.loads(repaired)
                except:
                    return None

        parsed = try_parse_and_repair(clean_text)
        if parsed and 'action' in parsed: return parsed

        for match in re.finditer(r'\{[^{}]*\}', clean_text):
            parsed = try_parse_and_repair(match.group(0))
            if parsed and 'action' in parsed: return parsed

        for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', clean_text):
            parsed = try_parse_and_repair(match.group(0))
            if parsed and 'action' in parsed: return parsed

        action_match = re.search(r'["\']action["\']:\s*["\'](\w+)["\']', clean_text)
        conf_match = re.search(r'["\']confidence["\']:\s*([0-9.]+)', clean_text)
        reason_match = re.search(r'["\']reasoning["\']:\s*["\'](.*?)["\']', clean_text, re.DOTALL)

        if action_match:
            return {
                'action': action_match.group(1).upper(),
                'confidence': float(conf_match.group(1)) if conf_match else 0.0,
                'reasoning': reason_match.group(1).replace("\n", " ") if reason_match else "Parsed via fallback"
            }

        self._parse_failures += 1
        print(f"🔥 JSON PARSE FAILED ({self._parse_failures}/{self._total_calls}). Raw Output snippet: {text[:150]}...")
        return {
            'action': 'KEEP', 
            'confidence': 0.0, 
            'reasoning': 'Parse Error: Could not extract valid JSON.'
        }

    def parse_failure_rate(self):
        if self._total_calls == 0:
            return 0.0
        return self._parse_failures / self._total_calls