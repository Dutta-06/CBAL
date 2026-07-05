class PromptBuilder:
    def __init__(self, template_dir=None):
        pass

    def build(self, error):
        """
        Balanced Prompting Strategy with Equal MERGE/KEEP Examples.
        """
        
        # --- CORE SYSTEM INSTRUCTION ---
        base_format = """
        ROLE: Expert in Speaker Diarization and Linguistic Analysis.
        
        OBJECTIVE: Determine if two consecutive segments should be MERGED (same speaker, continuous utterance) 
        or KEPT separate (different speakers OR natural turn boundary).
        
        Both MERGE and KEEP are equally valid outcomes - let linguistic evidence guide your decision.
        
        === DECISION CRITERIA ===
        
        MERGE Signals:
        ✓ Text A ends mid-phrase: "and", "but", "because", "to", "of", "that", "which"
        ✓ Text B starts lowercase (not capitalized)
        ✓ Text B completes A's grammatical structure
        ✓ NO period/question mark at end of Text A
        ✓ Small gap (< 1s) between segments
        
        KEEP Signals:
        ✗ Text A ends with [. ? !] and Text B starts capitalized
        ✗ Text B starts with reactive response: "Right", "Yeah", "Okay", "Yes", "No" (standalone)
        ✗ Question followed by Answer
        ✗ Text B uses "you/your" addressing A's speaker
        ✗ Text B evaluates/responds to A: "Exactly", "I agree", "I disagree"
        
        OUTPUT: {"action": "MERGE" or "KEEP", "confidence": 0.0-1.0, "reasoning": "brief explanation"}
        """

        # --- SCENARIO 1: FALSE SPLIT ---
        if error['type'] == 'false_split':
            parts = error['text'].split('[GAP]')
            text_a = parts[0].strip() if len(parts) > 0 else ""
            text_b = parts[1].strip() if len(parts) > 1 else ""
            
            return f"""
            {base_format}
            
            === FALSE SPLIT ANALYSIS ===
            TEXT A: "{text_a}"
            TEXT B: "{text_b}"
            GAP: {error['gap']:.2f}s
            
            ANALYZE: Does A end incomplete and B continue the thought?
            
            === BALANCED EXAMPLES (3 MERGE, 3 KEEP) ===
            
            MERGE #1:
            A: "We need to analyze the data and"
            B: "prepare the final report"
            → A ends with "and" (incomplete coordination)
            → B completes the compound predicate
            → MERGE | confidence: 0.92
            
            MERGE #2:
            A: "The system is designed to"
            B: "optimize performance automatically"
            → A ends with "to" (incomplete infinitive)
            → B provides the infinitive verb phrase
            → MERGE | confidence: 0.90
            
            MERGE #3:
            A: "I think that"
            B: "we should proceed with the plan"
            → A ends with "that" (incomplete complement clause)
            → B provides the subordinate clause
            → MERGE | confidence: 0.88
            
            KEEP #1:
            A: "I finished the analysis."
            B: "Right, now we can proceed."
            → A has period (complete sentence)
            → B starts with "Right" (reactive acknowledgment)
            → KEEP | confidence: 0.95
            
            KEEP #2:
            A: "Should we start the meeting now?"
            B: "Yes, everyone is here."
            → A is complete question
            → B is answer to that question
            → KEEP | confidence: 0.97
            
            KEEP #3:
            A: "The results are very promising."
            B: "I completely agree with that."
            → A is complete evaluative statement
            → B responds/agrees (separate turn)
            → KEEP | confidence: 0.93
            
            YOUR DECISION:
            """
        
        # --- SCENARIO 2: ACOUSTIC CONFUSION ---
        elif error['type'] == 'acoustic_confusion':
            parts = error['text'].split('[GAP]')
            text_a = parts[0].strip() if len(parts) > 0 else ""
            text_b = parts[1].strip() if len(parts) > 1 else ""

            return f"""
            {base_format}
            
            === ACOUSTIC CONFUSION ANALYSIS ===
            VOICE SIMILARITY: {error['similarity']:.2f} (high similarity, but check linguistic evidence)
            
            TEXT A: "{text_a}"
            TEXT B: "{text_b}"
            
            ANALYZE: Is this dialogue (two people) or continuation (one person)?
            
            === BALANCED EXAMPLES (3 MERGE, 3 KEEP) ===
            
            MERGE #1:
            A: "The analysis shows that"
            B: "the hypothesis is well supported"
            → A ends mid-sentence (incomplete "that" clause)
            → B completes the subordinate clause
            → No dialogue markers
            → MERGE | confidence: 0.87
            
            MERGE #2:
            A: "We implemented the changes and"
            B: "tested them thoroughly yesterday"
            → A ends with "and" (coordination incomplete)
            → B completes compound predicate
            → MERGE | confidence: 0.90
            
            MERGE #3:
            A: "The next step is to"
            B: "review all the documentation"
            → A ends with "to" (infinitive incomplete)
            → B provides infinitive phrase
            → MERGE | confidence: 0.85
            
            KEEP #1:
            A: "What do you think about this?"
            B: "I think it looks really good."
            → A asks question using "you" (addresses someone)
            → B answers in first person
            → KEEP | confidence: 0.98
            
            KEEP #2:
            A: "The deadline is next Friday."
            B: "Okay, we should finish by Thursday then."
            → A makes complete statement
            → B acknowledges with "Okay" and responds
            → KEEP | confidence: 0.94
            
            KEEP #3:
            A: "I disagree with that approach."
            B: "Well, I think it's the best option."
            → A states opinion
            → B counters with different view (disagreement)
            → KEEP | confidence: 0.96
            
            YOUR DECISION:
            """

        # --- SCENARIO 3: SHORT TURN ---
        elif error['type'] == 'short_turn_check':
            return f"""
            {base_format}
            
            === SHORT TURN ANALYSIS ===
            TEXT: "{error['text']}"
            DURATION: {error['duration']:.2f}s
            
            ANALYZE: Pure backchannel (no content) or substantive turn (has content)?
            
            === BALANCED EXAMPLES (3 MERGE, 3 KEEP) ===
            
            MERGE #1:
            Text: "Mhm"
            → Pure acknowledgment sound
            → No propositional content
            → MERGE | confidence: 0.88
            
            MERGE #2:
            Text: "Uh-huh"
            → Continuer/backchannel
            → Speaker maintains floor
            → MERGE | confidence: 0.85
            
            MERGE #3:
            Text: "Mm"
            → Minimal response
            → No semantic content
            → MERGE | confidence: 0.82
            
            KEEP #1:
            Text: "No"
            → Clear negative answer
            → Substantive response to question
            → KEEP | confidence: 0.95
            
            KEEP #2:
            Text: "Wait"
            → Imperative command
            → Takes conversational floor
            → KEEP | confidence: 0.93
            
            KEEP #3:
            Text: "Three"
            → Specific answer (likely to "how many?")
            → Provides new information
            → KEEP | confidence: 0.94
            
            YOUR DECISION:
            """
        
        return "Task unclear. Respond with KEEP."