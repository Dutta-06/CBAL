import os
import yaml
import torch
import json
from cbal.llm.gemma_agent import GemmaAgent
from cbal.llm.prompt_builder import PromptBuilder
from cbal.linguistic.context_builder import ContextBuilder
from cbal.linguistic.transcription import TranscriptLoader
from cbal.core.segmentation import load_rttm

# --- SETTINGS ---
MEETING_ID = "IS1009a"  # Using the file you verified has nested JSON
CONFIG_PATH = "configs/base_config.yaml"

def debug_run():
    # 1. Load Config
    print(f"⚙️ Loading config from {CONFIG_PATH}...")
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    
    # 2. Setup Components
    print(f"🤖 Initializing Gemma-3-4B-it...")
    # Ensure we use the model defined in config or override if needed
    agent = GemmaAgent(cfg['models']) 
    prompter = PromptBuilder()
    
    rttm_path = f"results/baseline_{MEETING_ID}.rttm"
    trans_path = f"results/transcripts/{MEETING_ID}.json"
    
    # Initialize Loader with the fixed logic for nested JSON
    loader = TranscriptLoader(trans_path)
    context_builder = ContextBuilder(loader)
    segments = load_rttm(rttm_path)

    print(f"\n🔍 Debugging decisions for {MEETING_ID}...")
    
    # 3. Simulate Errors (False Splits)
    # We check the first few segment pairs to see if the model merges them
    for i in range(len(segments) - 1):
        s1 = segments[i]
        s2 = segments[i+1]
        
        # --- THE FIX IS HERE ---
        # RTTM segments only have time. We must fetch text from the loader.
        text_a = loader.get_text(s1.start, s1.end)
        text_b = loader.get_text(s2.start, s2.end)
        
        # Calculate the gap between segments
        gap = s2.start - s1.end

        # Construct the error object with REAL text
        err = {
            'type': 'false_split',
            'indices': [i, i+1],
            'text': f"{text_a}\n[GAP {gap:.2f}s]\n{text_b}",
            'gap': gap,
            'duration': s1.duration + s2.duration 
        }

        # Build Prompt
        # This pulls the surrounding context to help Gemma understand the flow
        context_str = context_builder.build_context(segments, err['indices'])
        instruction = prompter.build(err)
        full_prompt = f"CONTEXT DIALOGUE:\n{context_str}\n\n{instruction}"

        # Visual Debug Output
        print("\n" + "="*60)
        print(f"STEP: Checking Segment {i} -> {i+1}")
        print(f"TEXT A: \"{text_a}\"") 
        print(f"TEXT B: \"{text_b}\"")
        print(f"GAP:    {gap:.2f}s")
        print("-" * 20)
        
        # 4. Get Raw Decision
        # This bypasses the merge engine to show you the raw "brain" of the agent
        try:
            raw_decision = agent.predict(full_prompt)
            print(f"🧠 AGENT RESPONSE: {raw_decision}")
            
            # 5. Check Guardrail Logic (Simulated)
            action = raw_decision.get('action', 'KEEP')
            conf = raw_decision.get('confidence', 0)
            threshold = cfg['thresholds']['min_confidence']
            
            if action == 'MERGE':
                if conf >= threshold:
                    print(f"✅ RESULT: WILL MERGE (Conf {conf} >= {threshold})")
                else:
                    print(f"❌ RESULT: REJECTED (Confidence {conf} too low)")
            else:
                print("⏭️ RESULT: KEEP (Model chose not to merge)")
                
        except Exception as e:
            print(f"🔥 Error during prediction: {e}")

        # Stop after 5 examples to keep output clean
        if i >= 5: 
            break

if __name__ == "__main__":
    debug_run()