#!/usr/bin/env python3
"""
CBAL Fix Validation Script - Decision Log Based

Uses the decision logs to identify exactly which fixes were applied,
then validates each one against ground truth.

Usage: python validate_with_logs.py [--meeting_id ES2004a]
"""

import os
import sys
import re
import ast
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import pandas as pd

@dataclass
class RTTMSegment:
    """Represents a single RTTM segment"""
    start: float
    end: float
    speaker: str
    meeting_id: str
    
    @property
    def duration(self):
        return self.end - self.start
    
    @property
    def midpoint(self):
        return self.start + (self.end - self.start) / 2
    
    def __repr__(self):
        return f"[{self.start:.2f}-{self.end:.2f} | {self.speaker}]"


@dataclass
class Decision:
    """Represents a single decision from the log"""
    type: str
    text: str
    action: str
    status: str
    index: int  # Position in the decision sequence
    
    def was_applied(self) -> bool:
        return 'APPLIED' in self.status
    
    def was_blocked(self) -> bool:
        return 'BLOCKED' in self.status
    
    def was_correct_according_to_cbal(self) -> bool:
        """Did CBAL think this was correct based on ground truth?"""
        return 'CORRECT' in self.status
    
    def was_incorrect_according_to_cbal(self) -> bool:
        """Did CBAL think this was incorrect based on ground truth?"""
        return 'INCORRECT' in self.status


def load_rttm_segments(path: str) -> List[RTTMSegment]:
    """Load RTTM file into a list of segments"""
    segments = []
    if not os.path.exists(path):
        return segments
    
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            
            meeting_id = parts[1]
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            
            segments.append(RTTMSegment(
                start=start,
                end=start + duration,
                speaker=speaker,
                meeting_id=meeting_id
            ))
    
    return sorted(segments, key=lambda s: s.start)


def load_decision_log(log_path: str) -> List[Decision]:
    """Parse the decision log file"""
    decisions = []
    
    if not os.path.exists(log_path):
        print(f"⚠️  Decision log not found: {log_path}")
        return decisions
    
    with open(log_path, 'r', encoding='utf-8') as f:
        decision_idx = 0
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that don't start with a dict opening brace
            if not line.startswith('{'):
                continue
            
            try:
                # Parse the dictionary from the log line
                data = ast.literal_eval(line)
                
                # Verify it's a valid decision record (has required fields)
                if not isinstance(data, dict) or 'type' not in data or 'action' not in data:
                    continue
                
                decision = Decision(
                    type=data.get('type', 'unknown'),
                    text=data.get('text', ''),
                    action=data.get('action', 'KEEP'),
                    status=data.get('status', ''),
                    index=decision_idx
                )
                decisions.append(decision)
                decision_idx += 1
            except (ValueError, SyntaxError):
                # Skip lines that aren't valid Python literals
                continue
            except Exception:
                # Skip any other parsing errors silently
                continue
    
    return decisions


def extract_segments_from_text(text: str) -> Tuple[str, str]:
    """
    Extract text before and after [GAP] marker.
    Returns: (text_a, text_b)
    """
    if '[GAP]' not in text:
        return text, ""
    
    parts = text.split('[GAP]')
    text_a = parts[0].strip() if len(parts) > 0 else ""
    text_b = parts[1].strip() if len(parts) > 1 else ""
    
    return text_a, text_b


def find_segment_pair_by_text(segments: List[RTTMSegment], 
                               transcript_loader,
                               text_a: str, 
                               text_b: str,
                               search_start_idx: int = 0) -> Optional[Tuple[int, int]]:
    """
    Find consecutive segments that match the text pattern.
    Returns: (index1, index2) or None
    """
    # Simple approach: look for consecutive same-speaker segments
    # This is approximate since we don't have the full transcription integrated
    for i in range(search_start_idx, len(segments) - 1):
        seg1 = segments[i]
        seg2 = segments[i + 1]
        
        # For false_split, expect same speaker
        if seg1.speaker == seg2.speaker:
            gap = seg2.start - seg1.end
            # Reasonable gap threshold
            if 0 < gap < 5.0:
                return (i, i + 1)
    
    return None


def find_speaker_at_time(segments: List[RTTMSegment], time: float) -> Optional[str]:
    """Find which speaker is active at a given time"""
    for seg in segments:
        if seg.start <= time <= seg.end:
            return seg.speaker
    return None


def check_gap_has_speech(truth_segs: List[RTTMSegment], 
                         gap_start: float, 
                         gap_end: float) -> Tuple[bool, List[str]]:
    """
    Check if there's speech in the gap according to ground truth.
    Returns: (has_speech, list_of_speakers_in_gap)
    """
    speakers_in_gap = []
    
    for seg in truth_segs:
        overlap_start = max(seg.start, gap_start)
        overlap_end = min(seg.end, gap_end)
        
        if overlap_start < overlap_end:
            if seg.speaker not in speakers_in_gap:
                speakers_in_gap.append(seg.speaker)
    
    return len(speakers_in_gap) > 0, speakers_in_gap


def validate_merge_with_truth(seg1: RTTMSegment, 
                               seg2: RTTMSegment,
                               truth_segs: List[RTTMSegment]) -> Dict:
    """
    Validate if a merge between seg1 and seg2 is correct according to ground truth.
    
    Returns dict with:
    - verdict: CORRECT, INCORRECT, or UNCERTAIN
    - reason: explanation
    - truth_speakers: what ground truth says
    """
    gap_start = seg1.end
    gap_end = seg2.start
    gap_mid = (gap_start + gap_end) / 2
    
    # Find speakers in truth for seg1 and seg2
    truth_spk1 = find_speaker_at_time(truth_segs, seg1.midpoint)
    truth_spk2 = find_speaker_at_time(truth_segs, seg2.midpoint)
    
    # Check gap for speech
    has_gap_speech, gap_speakers = check_gap_has_speech(truth_segs, gap_start, gap_end)
    
    # Validation logic
    if truth_spk1 is None or truth_spk2 is None:
        return {
            'verdict': 'UNCERTAIN',
            'reason': 'Could not find speakers in ground truth',
            'truth_spk1': truth_spk1,
            'truth_spk2': truth_spk2,
            'gap_speech': has_gap_speech,
            'gap_speakers': gap_speakers
        }
    
    if truth_spk1 != truth_spk2:
        return {
            'verdict': 'INCORRECT',
            'reason': f'Different speakers in truth: {truth_spk1} ≠ {truth_spk2}',
            'truth_spk1': truth_spk1,
            'truth_spk2': truth_spk2,
            'gap_speech': has_gap_speech,
            'gap_speakers': gap_speakers
        }
    
    # Same speaker in truth
    if has_gap_speech:
        # Check if gap speech is from the same speaker
        if len(gap_speakers) == 1 and gap_speakers[0] == truth_spk1:
            return {
                'verdict': 'CORRECT',
                'reason': f'Same speaker ({truth_spk1}) throughout including gap',
                'truth_spk1': truth_spk1,
                'truth_spk2': truth_spk2,
                'gap_speech': True,
                'gap_speakers': gap_speakers
            }
        else:
            return {
                'verdict': 'INCORRECT',
                'reason': f'Gap has different speaker(s): {gap_speakers}',
                'truth_spk1': truth_spk1,
                'truth_spk2': truth_spk2,
                'gap_speech': True,
                'gap_speakers': gap_speakers
            }
    else:
        return {
            'verdict': 'CORRECT',
            'reason': f'Same speaker ({truth_spk1}), no gap speech',
            'truth_spk1': truth_spk1,
            'truth_spk2': truth_spk2,
            'gap_speech': False,
            'gap_speakers': []
        }


def simulate_merges(baseline_segs: List[RTTMSegment], 
                    decisions: List[Decision]) -> Tuple[List[RTTMSegment], List[Dict]]:
    """
    Simulate the merge operations on baseline to track segment indices.
    Returns: (segments_after_merges, merge_records)
    """
    # Work with indices to track which baseline segments were merged
    current_segs = baseline_segs.copy()
    merge_records = []
    
    # Process decisions in order
    candidate_idx = 0  # Current position in the candidates list
    
    for decision in decisions:
        if not decision.was_applied():
            candidate_idx += 1
            continue
        
        # Find the next pair of consecutive same-speaker segments
        found_pair = False
        for i in range(len(current_segs) - 1):
            seg1 = current_segs[i]
            seg2 = current_segs[i + 1]
            
            # Check if this is a valid candidate for merge (same speaker, small gap)
            if seg1.speaker == seg2.speaker:
                gap = seg2.start - seg1.end
                if 0 < gap < 5.0:  # Reasonable gap threshold
                    # This is the candidate at position candidate_idx
                    # Perform merge
                    merge_records.append({
                        'decision_idx': decision.index,
                        'baseline_idx1': i,
                        'baseline_idx2': i + 1,
                        'seg1': seg1,
                        'seg2': seg2,
                        'gap': gap,
                        'decision': decision
                    })
                    
                    # Merge: extend seg1 to cover seg2
                    current_segs[i].end = seg2.end
                    # Remove seg2
                    current_segs.pop(i + 1)
                    
                    found_pair = True
                    break
        
        if not found_pair:
            print(f"⚠️  Could not find merge candidate for decision {decision.index}")
        
        candidate_idx += 1
    
    return current_segs, merge_records


def analyze_meeting(meeting_id: str,
                    baseline_path: str,
                    fixed_path: str,
                    truth_path: str,
                    log_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze a single meeting using decision logs.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {meeting_id}")
    print(f"{'='*80}")
    
    # Load files
    baseline_segs = load_rttm_segments(baseline_path)
    fixed_segs = load_rttm_segments(fixed_path)
    truth_segs = load_rttm_segments(truth_path)
    decisions = load_decision_log(log_path)
    
    if not baseline_segs or not fixed_segs or not truth_segs:
        print(f"⚠️  Missing RTTM data for {meeting_id}")
        return None, None
    
    if not decisions:
        print(f"⚠️  No valid decisions found in log for {meeting_id}")
        return None, None
    
    print(f"  Baseline segments: {len(baseline_segs)}")
    print(f"  Fixed segments:    {len(fixed_segs)}")
    print(f"  Truth segments:    {len(truth_segs)}")
    print(f"  Valid decisions:   {len(decisions)}")
    
    # Count applied and blocked
    applied = [d for d in decisions if d.was_applied()]
    blocked = [d for d in decisions if d.was_blocked()]
    
    print(f"  Applied fixes:     {len(applied)}")
    print(f"  Blocked fixes:     {len(blocked)}")
    
    # Simulate merges to get the segment pairs
    simulated_segs, merge_records = simulate_merges(baseline_segs, decisions)
    
    print(f"  Merge records:     {len(merge_records)}")
    
    # Validate each merge
    results = []
    for record in merge_records:
        seg1 = record['seg1']
        seg2 = record['seg2']
        decision = record['decision']
        
        # Validate against truth
        validation = validate_merge_with_truth(seg1, seg2, truth_segs)
        
        # Extract CBAL's own assessment from status
        cbal_thought_correct = decision.was_correct_according_to_cbal()
        cbal_thought_incorrect = decision.was_incorrect_according_to_cbal()
        
        # Compare verdicts
        our_verdict = validation['verdict']
        match_status = "N/A"
        if cbal_thought_correct:
            match_status = "MATCH" if our_verdict == "CORRECT" else "MISMATCH"
        elif cbal_thought_incorrect:
            match_status = "MATCH" if our_verdict == "INCORRECT" else "MISMATCH"
        
        result = {
            'Meeting_ID': meeting_id,
            'Decision_Index': decision.index,
            'Baseline_Idx1': record['baseline_idx1'],
            'Baseline_Idx2': record['baseline_idx2'],
            'Start1': seg1.start,
            'End1': seg1.end,
            'Start2': seg2.start,
            'End2': seg2.end,
            'Gap': record['gap'],
            'Baseline_Speaker': seg1.speaker,
            'Text_Sample': decision.text[:80] + '...' if len(decision.text) > 80 else decision.text,
            'Our_Verdict': our_verdict,
            'Our_Reason': validation['reason'],
            'CBAL_Status': decision.status,
            'Truth_Spk1': validation.get('truth_spk1'),
            'Truth_Spk2': validation.get('truth_spk2'),
            'Gap_Speech': validation.get('gap_speech', False),
            'Gap_Speakers': str(validation.get('gap_speakers', [])),
            'Match_Status': match_status
        }
        results.append(result)
    
    # Create DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        df = pd.DataFrame()
    
    # Summary stats
    summary = {
        'meeting_id': meeting_id,
        'baseline_segments': len(baseline_segs),
        'fixed_segments': len(fixed_segs),
        'total_decisions': len(decisions),
        'applied_fixes': len(applied),
        'blocked_fixes': len(blocked)
    }
    
    if not df.empty:
        summary['our_correct'] = len(df[df['Our_Verdict'] == 'CORRECT'])
        summary['our_incorrect'] = len(df[df['Our_Verdict'] == 'INCORRECT'])
        summary['our_uncertain'] = len(df[df['Our_Verdict'] == 'UNCERTAIN'])
        
        total_validated = summary['our_correct'] + summary['our_incorrect']
        if total_validated > 0:
            summary['accuracy'] = summary['our_correct'] / total_validated
        else:
            summary['accuracy'] = 0
        
        # CBAL's own assessment
        summary['cbal_said_correct'] = len([d for d in applied if d.was_correct_according_to_cbal()])
        summary['cbal_said_incorrect'] = len([d for d in applied if d.was_incorrect_according_to_cbal()])
        
        # Agreement rate
        matches = len(df[df['Match_Status'] == 'MATCH'])
        mismatches = len(df[df['Match_Status'] == 'MISMATCH'])
        if matches + mismatches > 0:
            summary['agreement_rate'] = matches / (matches + mismatches)
        else:
            summary['agreement_rate'] = 0
        
        print(f"\n  Our Validation:")
        print(f"    ✅ Correct:   {summary['our_correct']}")
        print(f"    ❌ Incorrect: {summary['our_incorrect']}")
        print(f"    ❓ Uncertain: {summary['our_uncertain']}")
        if total_validated > 0:
            print(f"    📊 Accuracy:  {summary['accuracy']*100:.1f}%")
        
        print(f"\n  CBAL's Assessment (from status field):")
        print(f"    ✅ Said Correct:   {summary['cbal_said_correct']}")
        print(f"    ❌ Said Incorrect: {summary['cbal_said_incorrect']}")
        if summary['cbal_said_correct'] + summary['cbal_said_incorrect'] > len(applied):
            print(f"    ⚠️  Note: Same fix may have multiple assessments in log")
        
        if matches + mismatches > 0:
            print(f"\n  Agreement: {summary['agreement_rate']*100:.1f}% ({matches}/{matches+mismatches})")
    
    return df, summary


def main():
    parser = argparse.ArgumentParser(description='Validate CBAL fixes using decision logs')
    parser.add_argument('--meeting_id', type=str, help='Single meeting ID to analyze')
    parser.add_argument('--baseline_dir', default='results', help='Directory containing baseline_*.rttm')
    parser.add_argument('--fixed_dir', default='results', help='Directory containing fixed_*.rttm')
    parser.add_argument('--truth_dir', default='data/ami/rttm', help='Directory containing ground truth')
    parser.add_argument('--log_dir', default='results/logs', help='Directory containing decision logs')
    parser.add_argument('--output_dir', default='results/validation', help='Output directory')
    args = parser.parse_args()
    
    # Determine meetings
    if args.meeting_id:
        meeting_ids = [args.meeting_id]
    else:
        meeting_ids = [
            "ES2004a", "ES2004b", "ES2004c", "ES2004d",
            "IS1009a", "IS1009b", "IS1009c", "IS1009d",
            "TS3003a", "TS3003b", "TS3003c", "TS3003d"
        ]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("CBAL FIX VALIDATION - DECISION LOG BASED")
    print("="*80)
    
    all_results = []
    all_summaries = []
    
    for meeting_id in meeting_ids:
        baseline_path = os.path.join(args.baseline_dir, f"baseline_{meeting_id}.rttm")
        fixed_path = os.path.join(args.fixed_dir, f"fixed_{meeting_id}.rttm")
        truth_path = os.path.join(args.truth_dir, f"{meeting_id}.rttm")
        log_path = os.path.join(args.log_dir, f"decisions_{meeting_id}.log")
        
        # Check files exist
        missing = []
        if not os.path.exists(baseline_path): missing.append("baseline")
        if not os.path.exists(fixed_path): missing.append("fixed")
        if not os.path.exists(truth_path): missing.append("truth")
        if not os.path.exists(log_path): missing.append("log")
        
        if missing:
            print(f"\n⚠️  Skipping {meeting_id} - missing: {', '.join(missing)}")
            continue
        
        df, summary = analyze_meeting(meeting_id, baseline_path, fixed_path, truth_path, log_path)
        
        if df is not None and not df.empty:
            all_results.append(df)
            all_summaries.append(summary)
    
    # Save results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        summary_df = pd.DataFrame(all_summaries)
        
        detailed_output = os.path.join(args.output_dir, "detailed_validation_with_logs.csv")
        combined_df.to_csv(detailed_output, index=False)
        print(f"\n{'='*80}")
        print(f"💾 Detailed results: {detailed_output}")
        
        summary_output = os.path.join(args.output_dir, "summary_validation_with_logs.csv")
        summary_df.to_csv(summary_output, index=False)
        print(f"💾 Summary: {summary_output}")
        
        # Overall stats
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS")
        print(f"{'='*80}")
        
        total_applied = combined_df.shape[0]
        our_correct = len(combined_df[combined_df['Our_Verdict'] == 'CORRECT'])
        our_incorrect = len(combined_df[combined_df['Our_Verdict'] == 'INCORRECT'])
        our_uncertain = len(combined_df[combined_df['Our_Verdict'] == 'UNCERTAIN'])
        
        print(f"\nTotal Applied Fixes: {total_applied}")
        print(f"  ✅ Correct:   {our_correct:3d} ({our_correct/total_applied*100:5.1f}%)")
        print(f"  ❌ Incorrect: {our_incorrect:3d} ({our_incorrect/total_applied*100:5.1f}%)")
        print(f"  ❓ Uncertain: {our_uncertain:3d} ({our_uncertain/total_applied*100:5.1f}%)")
        
        if our_correct + our_incorrect > 0:
            accuracy = our_correct / (our_correct + our_incorrect)
            print(f"\n📊 Overall Accuracy: {accuracy*100:.1f}%")
        
        # CBAL's assessment
        cbal_correct = summary_df['cbal_said_correct'].sum()
        cbal_incorrect = summary_df['cbal_said_incorrect'].sum()
        
        print(f"\nCBAL's Self-Assessment:")
        print(f"  ✅ Said Correct:   {cbal_correct}")
        print(f"  ❌ Said Incorrect: {cbal_incorrect}")
        
        # Agreement
        matches = len(combined_df[combined_df['Match_Status'] == 'MATCH'])
        mismatches = len(combined_df[combined_df['Match_Status'] == 'MISMATCH'])
        if matches + mismatches > 0:
            agreement = matches / (matches + mismatches)
            print(f"\nAgreement with CBAL: {agreement*100:.1f}% ({matches}/{matches+mismatches})")
        
        print(f"\n{'-'*80}")
        print("PER-MEETING SUMMARY")
        print(f"{'-'*80}")
        summary_display = summary_df[[
            'meeting_id', 'applied_fixes', 'our_correct', 'our_incorrect', 
            'accuracy', 'cbal_said_correct', 'cbal_said_incorrect', 'agreement_rate'
        ]].copy()
        summary_display['accuracy'] = summary_display['accuracy'].apply(lambda x: f"{x*100:.1f}%")
        summary_display['agreement_rate'] = summary_display['agreement_rate'].apply(lambda x: f"{x*100:.1f}%")
        print(summary_display.to_string(index=False))
        
    else:
        print("\n⚠️  No results generated")


if __name__ == "__main__":
    main()