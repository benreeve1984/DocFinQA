#!/usr/bin/env python3
"""
Test script to verify the DocFinQA evaluation setup

This script performs basic tests to ensure all components are working correctly.
"""

import json
import os
import sys
from pathlib import Path

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    
    try:
        from dotenv import load_dotenv
        print("✓ dotenv import successful")
    except ImportError as e:
        print(f"✗ dotenv import failed: {e}")
        return False
        
    try:
        import openai
        print("✓ openai import successful")
    except ImportError as e:
        print(f"✗ openai import failed: {e}")
        return False
        
    try:
        from evals.scorer import score_response, calculate_aggregate_scores
        print("✓ scorer import successful")
    except ImportError as e:
        print(f"✗ scorer import failed: {e}")
        return False
        
    try:
        from evals.docfinqa_eval import DocFinQAEvaluator, TestCase
        print("✓ docfinqa_eval import successful")
    except ImportError as e:
        print(f"✗ docfinqa_eval import failed: {e}")
        return False
        
    return True

def test_env_file():
    """Test that .env file exists and has required variables"""
    print("\nTesting environment configuration...")
    
    env_file = Path('.env')
    if not env_file.exists():
        print(f"✗ .env file not found. Please copy .env-template to .env and configure it.")
        return False
    
    print("✓ .env file exists")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("✗ OPENAI_API_KEY not set in .env file")
        return False
    elif api_key == 'your_openai_api_key_here':
        print("✗ OPENAI_API_KEY still set to placeholder value")
        return False
    else:
        print("✓ OPENAI_API_KEY is configured")
        
    return True

def test_data_file():
    """Test that test data file exists and is properly formatted"""
    print("\nTesting test data file...")
    
    data_file = Path('data/test-data-sample.json')
    if not data_file.exists():
        print(f"✗ Test data file not found: {data_file}")
        return False
        
    print("✓ Test data file exists")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("✗ Test data should be a list")
            return False
            
        if len(data) == 0:
            print("✗ Test data is empty")
            return False
            
        print(f"✓ Test data loaded successfully ({len(data)} records)")
        
        # Check first record structure
        first_record = data[0]
        required_keys = ['Context', 'Question', 'Answer']
        missing_keys = [key for key in required_keys if key not in first_record]
        
        if missing_keys:
            print(f"✗ Missing required keys in test data: {missing_keys}")
            return False
            
        print("✓ Test data format is correct")
        
        # Check context length
        context_length = len(first_record['Context'])
        print(f"✓ Sample context length: {context_length:,} characters")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in test data file: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading test data file: {e}")
        return False

def test_scorer():
    """Test the scorer functionality"""
    print("\nTesting scorer functionality...")
    
    try:
        from evals.scorer import score_response
        
        # Test basic scoring
        predicted = "The company's revenue increased by 15% to $100 million in Q4."
        expected = "Revenue grew 15% to $100M in Q4."
        
        score = score_response(predicted, expected)
        
        print(f"✓ Scoring test successful:")
        print(f"  - Exact match: {score.exact_match}")
        print(f"  - Normalized exact match: {score.normalized_exact_match}")
        print(f"  - F1 score: {score.f1_score:.3f}")
        print(f"  - Semantic similarity: {score.semantic_similarity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Scorer test failed: {e}")
        return False

def test_openai_connection():
    """Test basic OpenAI API connection"""
    print("\nTesting OpenAI API connection...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from openai import OpenAI
        
        client = OpenAI()
        
        # Try to list models (simple API call)
        models = client.models.list()
        print("✓ OpenAI API connection successful")
        print(f"✓ Available models: {len(models.data)} models found")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenAI API connection failed: {e}")
        print("  Make sure your API key is correct and you have internet access")
        return False

def main():
    """Run all tests"""
    print("DocFinQA Evaluation Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_env_file,
        test_data_file,
        test_scorer,
        test_openai_connection
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready for evaluation.")
        print("\nTo run a quick evaluation test:")
        print("python evals/docfinqa_eval.py --limit 1")
    else:
        print("❌ Some tests failed. Please fix the issues above before running evaluations.")
        sys.exit(1)

if __name__ == '__main__':
    main() 