#!/usr/bin/env python3
"""
Scorer module for DocFinQA evaluation

=== WHAT THIS MODULE DOES ===
This module evaluates how well AI responses match expected answers in financial Q&A.
The main focus is on numerical accuracy - did the AI extract the right number?

=== KEY CONCEPTS FOR BEGINNERS ===
- Numerical Accuracy: Our main metric - extracts numbers and compares them
- Tolerances: We allow small differences (±0.5 absolute, ±2% relative) for rounding
- FINAL Tag: We prefer answers in "FINAL: 123.45" format but handle other formats too
- Fallback Logic: If no "FINAL:" tag, we extract the last number in the response

=== WHY NUMERICAL ACCURACY? ===
Financial Q&A is about numbers, not text similarity. Traditional metrics like F1 or BLEU
don't make sense when comparing "21.48" vs "21.50" - we need domain-specific scoring.

=== USAGE EXAMPLES ===
    from scorer import score_response
    
    # Score a response
    result = score_response("FINAL: 21.48", "21.48", "What was Q3 revenue?")
    print(f"Accuracy: {result.numerical_accuracy}")  # 1.0 (perfect)
    
    # Also handles tolerance
    result = score_response("FINAL: 21.47", "21.48", "What was Q3 revenue?") 
    print(f"Accuracy: {result.numerical_accuracy}")  # 1.0 (within tolerance)
"""

# === PYTHON IMPORTS ===
import re           # For regular expressions (pattern matching)
import math         # For numerical comparisons with tolerance
import string       # For text processing
from typing import Dict, List, Any, Optional  # Type hints for better code
from dataclasses import dataclass             # For creating data structures
import difflib      # For text similarity calculations
from collections import Counter               # For counting tokens

# === PATTERN MATCHING FOR FINANCIAL ANSWERS ===
# This regular expression finds numbers after "FINAL:" in any case
# Example: "FINAL: 21.48" or "final: -100.5" or "Final: 1,234.56"
_FINAL_RE = re.compile(r'FINAL:\s*([-+]?\d*\.?\d+)', re.I)  # re.I = case insensitive

# === DATA STRUCTURE FOR SCORING RESULTS ===

@dataclass
class ScoreResult:
    """
    Represents the score for a single response
    
    This contains all the scoring information for one question-answer pair.
    The most important field is numerical_accuracy (0.0 to 1.0).
    
    Example:
        If we asked "What was Q3 revenue?" and expected "100":
        - numerical_accuracy = 1.0 if AI said "FINAL: 100" 
        - numerical_accuracy = 0.0 if AI said "I don't know"
        - extracted_number = 100.0 (what we found in AI response)
        - expected_number = 100.0 (what we were looking for)
        - found_final_tag = True (AI used proper "FINAL:" format)
    """
    # === PRIMARY METRIC (what we care about most) ===
    numerical_accuracy: float        # 0.0 to 1.0 - did numbers match within tolerance?
    
    # === ADDITIONAL TEXT-BASED METRICS (for analysis) ===
    exact_match: bool               # Did text match exactly? (rarely useful)
    normalized_exact_match: bool    # Did text match after cleaning? 
    f1_score: float                 # Token overlap score (not great for numbers)
    bleu_score: float               # Translation-style score (not great for numbers)
    semantic_similarity: float      # Text similarity (not great for numbers)
    
    # === EXTRACTION INFORMATION ===
    extracted_number: Optional[float]  # Number we found in AI response (or None)
    expected_number: Optional[float]   # Number we expected (or None)
    found_final_tag: bool              # Did AI use "FINAL: X" format?
    details: Dict[str, Any]            # Extra debugging information

# === NUMBER EXTRACTION FUNCTION ===
# This is the heart of our financial Q&A scoring

def _extract_number(txt: str | None):
    """
    Extract a number from AI response text - this is where the magic happens!
    
    Strategy:
    1. First, look for "FINAL: 123.45" format (preferred)
    2. If not found, take the last number anywhere in the text (fallback)
    3. If no numbers at all, return None
    
    This handles cases where the AI:
    - Uses proper format: "FINAL: 21.48" 
    - Forgets the tag: "The revenue was 21.48 million"
    - Has multiple numbers: "Revenue grew from 20.5 to 21.48" (takes 21.48)
    
    Args:
        txt: Text to extract number from (AI's response)
        
    Returns:
        Extracted number as float, or None if no number found
        
    Examples:
        _extract_number("FINAL: 21.48") -> 21.48
        _extract_number("Revenue was $21.48M") -> 21.48  
        _extract_number("I don't know") -> None
    """
    if not txt:  # Handle None or empty string
        return None
        
    # === STEP 1: LOOK FOR "FINAL: NUMBER" PATTERN (PREFERRED) ===
    # Find all numbers that come after "FINAL:"
    m = _FINAL_RE.findall(txt)  # Returns list of number strings
    if m:
        # Take the last "FINAL:" number (in case there are multiple)
        return float(m[-1])
        
    # === STEP 2: FALLBACK - FIND ANY NUMBERS (LAST ONE) ===
    # Pattern matches: 123, -456, 78.9, +1.23, etc.
    nums = re.findall(r'[-+]?\d*\.?\d+', txt)
    if nums:
        # Take the last number found (often the final answer)
        return float(nums[-1])
        
    # === STEP 3: NO NUMBERS FOUND ===
    return None

# === NUMERICAL ACCURACY SCORING FUNCTION ===
# This is our main scoring function for financial Q&A

def numerical_accuracy_score(output: str = None,
                           expected: str = None,
                           abs_tol: float = 0.5,
                           rel_tol: float = 0.02) -> tuple[float, Optional[float], Optional[float], bool]:
    """
    Calculate numerical accuracy score for financial Q&A - our main metric!
    
    This function determines if the AI got the right number within reasonable tolerance.
    Financial numbers should be exact, but we allow small rounding differences.
    
    How it works:
    1. Extract number from AI's response (prefer "FINAL: X" format)
    2. Extract number from expected answer
    3. Compare them with tolerance for rounding errors
    4. Return 1.0 if match, 0.0 if no match
    
    Args:
        output: AI model's response text (e.g., "FINAL: 21.48")
        expected: Expected answer (e.g., "21.5" or "21.48")
        abs_tol: Absolute tolerance (default 0.5) - numbers within ±0.5 are considered equal
        rel_tol: Relative tolerance (default 0.02 = 2%) - 2% difference allowed
        
    Returns:
        Tuple of (score, extracted_number, expected_number, found_final_tag)
        - score: 1.0 if numbers match within tolerance, 0.0 otherwise
        - extracted_number: Number found in AI response (or None)
        - expected_number: Number found in expected answer (or None) 
        - found_final_tag: True if AI used "FINAL:" format
        
    Examples:
        numerical_accuracy_score("FINAL: 21.48", "21.48") -> (1.0, 21.48, 21.48, True)
        numerical_accuracy_score("FINAL: 21.47", "21.48") -> (1.0, 21.47, 21.48, True)  # within tolerance
        numerical_accuracy_score("FINAL: 25.0", "21.48")  -> (0.0, 25.0, 21.48, True)   # too different
        numerical_accuracy_score("I don't know", "21.48") -> (0.0, None, 21.48, False)  # no number
    """
    # === STEP 1: EXTRACT NUMBERS FROM BOTH TEXTS ===
    extracted = _extract_number(output)        # Number from AI response
    expected_num = _extract_number(str(expected))  # Number from expected answer
    found_final = bool(_FINAL_RE.search(output or ""))  # Did AI use "FINAL:" format?
    
    # === STEP 2: CHECK IF WE FOUND NUMBERS IN BOTH ===
    if extracted is None or expected_num is None:
        # If we can't find a number in either text, it's a failure
        return 0.0, extracted, expected_num, found_final
    
    # === STEP 3: COMPARE NUMBERS WITH TOLERANCE ===
    # Use Python's math.isclose() function which handles both absolute and relative tolerance
    # This means 21.47 and 21.48 are "close enough" but 21.0 and 25.0 are not
    numbers_match = math.isclose(extracted, expected_num, abs_tol=abs_tol, rel_tol=rel_tol)
    score = 1.0 if numbers_match else 0.0
    
    return score, extracted, expected_num, found_final

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def calculate_f1_score(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    """
    Calculate F1 score between predicted and gold tokens
    
    Args:
        pred_tokens: Predicted tokens
        gold_tokens: Gold standard tokens
        
    Returns:
        F1 score
    """
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
        
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    
    # Calculate overlap
    overlap = sum((pred_counter & gold_counter).values())
    
    if overlap == 0:
        return 0.0
        
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def calculate_bleu_score(pred_text: str, gold_text: str) -> float:
    """
    Calculate a simple BLEU-like score
    
    Args:
        pred_text: Predicted text
        gold_text: Gold standard text
        
    Returns:
        BLEU-like score
    """
    pred_tokens = normalize_text(pred_text).split()
    gold_tokens = normalize_text(gold_text).split()
    
    if not pred_tokens or not gold_tokens:
        return 0.0
        
    # Simple unigram BLEU
    return calculate_f1_score(pred_tokens, gold_tokens)

def calculate_semantic_similarity(pred_text: str, gold_text: str) -> float:
    """
    Calculate semantic similarity (placeholder implementation)
    
    Args:
        pred_text: Predicted text
        gold_text: Gold standard text
        
    Returns:
        Semantic similarity score
    """
    # TODO: Implement with sentence transformers or similar
    # For now, use a simple string similarity
    
    normalized_pred = normalize_text(pred_text)
    normalized_gold = normalize_text(gold_text)
    
    if not normalized_pred and not normalized_gold:
        return 1.0
    if not normalized_pred or not normalized_gold:
        return 0.0
        
    # Use SequenceMatcher for string similarity
    similarity = difflib.SequenceMatcher(None, normalized_pred, normalized_gold).ratio()
    return similarity

def score_response(predicted: str, expected: str, question: str = "") -> ScoreResult:
    """
    Score a single response against the expected answer using multiple metrics.
    
    The primary metric for financial Q&A is numerical_accuracy, which compares
    extracted numbers with appropriate tolerances.
    
    Args:
        predicted: Model's predicted response
        expected: Expected/gold standard answer
        question: Original question (for context)
        
    Returns:
        ScoreResult object with various scores
    """
    # Primary metric: numerical accuracy (best for financial Q&A)
    num_score, extracted_num, expected_num, found_final = numerical_accuracy_score(predicted, expected)
    
    # Additional text-based metrics (for completeness)
    # Exact match
    exact_match = predicted.strip() == expected.strip()
    
    # Normalized exact match
    normalized_pred = normalize_text(predicted)
    normalized_gold = normalize_text(expected)
    normalized_exact_match = normalized_pred == normalized_gold
    
    # F1 score
    pred_tokens = normalized_pred.split()
    gold_tokens = normalized_gold.split()
    f1_score = calculate_f1_score(pred_tokens, gold_tokens)
    
    # BLEU score
    bleu_score = calculate_bleu_score(predicted, expected)
    
    # Semantic similarity
    semantic_similarity = calculate_semantic_similarity(predicted, expected)
    
    # Additional details
    details = {
        'predicted_length': len(predicted),
        'expected_length': len(expected),
        'predicted_tokens': len(pred_tokens),
        'expected_tokens': len(gold_tokens),
        'normalized_predicted': normalized_pred,
        'normalized_expected': normalized_gold,
        'contains_final_tag': found_final,
        'numerical_difference': abs(extracted_num - expected_num) if (extracted_num is not None and expected_num is not None) else None
    }
    
    return ScoreResult(
        numerical_accuracy=num_score,
        exact_match=exact_match,
        normalized_exact_match=normalized_exact_match,
        f1_score=f1_score,
        bleu_score=bleu_score,
        semantic_similarity=semantic_similarity,
        extracted_number=extracted_num,
        expected_number=expected_num,
        found_final_tag=found_final,
        details=details
    )

def calculate_aggregate_scores(score_results: List[ScoreResult]) -> Dict[str, float]:
    """
    Calculate aggregate scores from a list of individual score results
    
    Args:
        score_results: List of ScoreResult objects
        
    Returns:
        Dictionary of aggregate scores
    """
    if not score_results:
        return {}
        
    total_cases = len(score_results)
    
    # Primary metric
    avg_numerical_accuracy = sum(sr.numerical_accuracy for sr in score_results) / total_cases
    
    # Additional metrics
    exact_match_rate = sum(1 for sr in score_results if sr.exact_match) / total_cases
    normalized_exact_match_rate = sum(1 for sr in score_results if sr.normalized_exact_match) / total_cases
    avg_f1_score = sum(sr.f1_score for sr in score_results) / total_cases
    avg_bleu_score = sum(sr.bleu_score for sr in score_results) / total_cases
    avg_semantic_similarity = sum(sr.semantic_similarity for sr in score_results) / total_cases
    
    # Success rates for numerical extraction
    successful_extractions = sum(1 for sr in score_results if sr.extracted_number is not None)
    extraction_success_rate = successful_extractions / total_cases
    
    final_tag_usage = sum(1 for sr in score_results if sr.found_final_tag)
    final_tag_rate = final_tag_usage / total_cases
    
    return {
        # Primary metric for financial Q&A
        'numerical_accuracy': avg_numerical_accuracy,
        
        # Secondary metrics
        'exact_match_rate': exact_match_rate,
        'normalized_exact_match_rate': normalized_exact_match_rate,
        'average_f1_score': avg_f1_score,
        'average_bleu_score': avg_bleu_score,
        'average_semantic_similarity': avg_semantic_similarity,
        
        # Extraction statistics
        'extraction_success_rate': extraction_success_rate,
        'final_tag_usage_rate': final_tag_rate,
        
        # Metadata
        'total_cases': total_cases,
        'successful_extractions': successful_extractions
    }

def detailed_analysis(score_results: List[ScoreResult]) -> Dict[str, Any]:
    """
    Perform detailed analysis of the score results
    
    Args:
        score_results: List of ScoreResult objects
        
    Returns:
        Dictionary with detailed analysis
    """
    if not score_results:
        return {}
        
    # Numerical accuracy statistics
    num_scores = [sr.numerical_accuracy for sr in score_results]
    perfect_scores = sum(1 for score in num_scores if score == 1.0)
    failed_scores = sum(1 for score in num_scores if score == 0.0)
    
    # Extraction analysis
    successful_extractions = [sr for sr in score_results if sr.extracted_number is not None]
    failed_extractions = [sr for sr in score_results if sr.extracted_number is None]
    
    # Number differences for successful extractions
    differences = []
    for sr in successful_extractions:
        if sr.expected_number is not None:
            diff = abs(sr.extracted_number - sr.expected_number)
            differences.append(diff)
    
    analysis = {
        'numerical_performance': {
            'perfect_matches': perfect_scores,
            'failed_matches': failed_scores,
            'accuracy_rate': perfect_scores / len(score_results),
            'avg_difference': sum(differences) / len(differences) if differences else 0,
            'max_difference': max(differences) if differences else 0
        },
        'extraction_analysis': {
            'successful_extractions': len(successful_extractions),
            'failed_extractions': len(failed_extractions),
            'final_tag_usage': sum(1 for sr in score_results if sr.found_final_tag),
            'extraction_failure_rate': len(failed_extractions) / len(score_results)
        },
        'length_analysis': {
            'avg_predicted_length': sum(sr.details['predicted_length'] for sr in score_results) / len(score_results),
            'avg_expected_length': sum(sr.details['expected_length'] for sr in score_results) / len(score_results),
            'avg_predicted_tokens': sum(sr.details['predicted_tokens'] for sr in score_results) / len(score_results),
            'avg_expected_tokens': sum(sr.details['expected_tokens'] for sr in score_results) / len(score_results)
        }
    }
    
    return analysis

# Legacy function for backward compatibility with your original scorer
def handler(output: str = None,
            expected: str = None,
            **kwargs) -> float:
    """
    Legacy handler function that matches your original scorer interface.
    Returns 1.0 if the two numbers match (±0.01 absolute or ±0.1 % relative).
    """
    return numerical_accuracy_score(output, expected)[0]

# Example usage
if __name__ == '__main__':
    # Test with financial examples
    test_cases = [
        ("The final answer is 21.48", "21.48"),
        ("FINAL: 15.2", "15.20"),
        ("Revenue increased to $100.5 million", "100.5"),
        ("The growth rate was 7.8%", "7.8"),
        ("FINAL: 42.0", "42.1"),  # Should pass with tolerance
        ("No numerical answer", "25.0"),  # Should fail
    ]
    
    print("Testing Financial Q&A Scorer:")
    print("=" * 50)
    
    all_results = []
    for predicted, expected in test_cases:
        score = score_response(predicted, expected)
        all_results.append(score)
        
        print(f"Predicted: {predicted}")
        print(f"Expected: {expected}")
        print(f"Numerical Accuracy: {score.numerical_accuracy}")
        print(f"Extracted: {score.extracted_number}")
        print(f"Expected Num: {score.expected_number}")
        print(f"Found FINAL: {score.found_final_tag}")
        print("-" * 30)
    
    # Aggregate results
    aggregates = calculate_aggregate_scores(all_results)
    print(f"\nAggregate Results:")
    print(f"Numerical Accuracy: {aggregates['numerical_accuracy']:.3f}")
    print(f"Extraction Success Rate: {aggregates['extraction_success_rate']:.3f}")
    print(f"FINAL Tag Usage: {aggregates['final_tag_usage_rate']:.3f}") 