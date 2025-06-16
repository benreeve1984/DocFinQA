#!/usr/bin/env python3
"""
Test Batch Evaluation Script for DocFinQA

This is a test version that runs just 1 test case per mode to verify
the batch evaluation system works correctly before running the full 100 tests.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import json

def run_evaluation(mode_name, mode_args, limit=1):
    """
    Run a single evaluation mode with resume capability
    
    Args:
        mode_name: Descriptive name for the mode (used in filename)
        mode_args: List of command line arguments for the evaluation
        limit: Number of test cases to run
    
    Returns:
        (success, output_file, duration, completed_cases)
    """
    # Generate timestamped filename (but check for existing file first)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_file = f"results/docfinqa_eval_{mode_name}_{timestamp}.json"
    
    # Check for existing partial files with same mode name from today
    existing_file = find_existing_partial_file(mode_name)
    
    if existing_file:
        output_file = existing_file
        print(f"🔄 Found existing partial results: {existing_file}")
        print(f"   Will resume from where it left off...")
    else:
        output_file = base_output_file
        print(f"🚀 Starting fresh {mode_name} evaluation...")
    
    # Build the command
    cmd = [
        "python", "evals/docfinqa_eval.py",
        "--limit", str(limit),
        "--output", output_file
    ] + mode_args
    
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Output: {output_file}")
    
    start_time = time.time()
    
    try:
        # Run the evaluation
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            # Check how many cases were actually completed
            completed_cases = count_completed_cases(output_file)
            
            if completed_cases >= limit:
                print(f"✅ {mode_name} completed successfully in {duration:.1f}s")
                print(f"   All {completed_cases} test cases completed")
            else:
                print(f"⚠️  {mode_name} partially completed in {duration:.1f}s")
                print(f"   Completed {completed_cases}/{limit} test cases")
                print(f"   Results saved to: {output_file}")
                print(f"   ⭐ Run the batch script again to resume from this point")
                
            return True, output_file, duration, completed_cases
        else:
            print(f"❌ {mode_name} failed!")
            print(f"   Error: {result.stderr}")
            
            # Check if we have any partial results
            completed_cases = count_completed_cases(output_file) if os.path.exists(output_file) else 0
            return False, output_file if completed_cases > 0 else None, duration, completed_cases
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ {mode_name} failed with exception: {e}")
        
        # Check if we have any partial results
        completed_cases = count_completed_cases(output_file) if os.path.exists(output_file) else 0
        return False, output_file if completed_cases > 0 else None, duration, completed_cases

def find_existing_partial_file(mode_name):
    """
    Find existing partial result files for the given mode from today
    
    Args:
        mode_name: Name of the evaluation mode
        
    Returns:
        Path to existing file or None
    """
    if not os.path.exists('results'):
        return None
        
    today = datetime.now().strftime('%Y%m%d')
    pattern = f"docfinqa_eval_{mode_name}_{today}_*.json"
    
    import glob
    matching_files = glob.glob(f"results/{pattern}")
    
    if matching_files:
        # Return the most recent file
        return max(matching_files, key=os.path.getctime)
    
    return None

def count_completed_cases(output_file):
    """
    Count how many test cases have been completed in a results file
    
    Args:
        output_file: Path to results file
        
    Returns:
        Number of completed test cases
    """
    try:
        if not os.path.exists(output_file):
            return 0
            
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return len(data.get('results', []))
        
    except Exception:
        return 0

def main():
    """Run test batch evaluation across all modes"""
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("🧪 DocFinQA Test Batch Evaluation")
    print("=" * 50)
    print("Running 1 test for each of 4 evaluation modes...")
    print()
    
    # Define the evaluation modes
    modes = [
        {
            "name": "closed_book",
            "description": "Closed Book (No Context)",
            "args": ["--evaluation-mode", "closed_book"]
        },
        {
            "name": "file_search_no_rewrite", 
            "description": "File Search (No Query Rewriting)",
            "args": ["--evaluation-mode", "file_search"]
        },
        {
            "name": "file_search_with_rewrite",
            "description": "File Search (With Query Rewriting)", 
            "args": ["--evaluation-mode", "file_search", "--rewrite-queries"]
        },
        {
            "name": "context_window",
            "description": "Context Window (Full Context in Prompt)",
            "args": ["--evaluation-mode", "context_window"]
        }
    ]
    
    # Track results
    results = []
    total_start_time = time.time()
    
    # Run each evaluation mode
    for i, mode in enumerate(modes, 1):
        print(f"📋 Mode {i}/4: {mode['description']}")
        print("-" * 30)
        
        success, output_file, duration, completed_cases = run_evaluation(
            mode_name=mode['name'],
            mode_args=mode['args'],
            limit=1  # Just 1 test case for testing
        )
        
        results.append({
            'mode': mode['name'],
            'description': mode['description'],
            'success': success,
            'output_file': output_file,
            'duration': duration,
            'completed_cases': completed_cases
        })
        
        print()
        
        # Add a brief pause between modes (shorter for testing)
        if i < len(modes):
            print("⏱️  Pausing 5 seconds before next mode...")
            time.sleep(5)
            print()
    
    # Print final summary
    total_duration = time.time() - total_start_time
    
    print("🏁 Test Batch Evaluation Complete!")
    print("=" * 50)
    print(f"Total time: {total_duration:.1f} seconds")
    print()
    
    successful_modes = sum(1 for r in results if r['success'])
    print(f"✅ Successful modes: {successful_modes}/{len(modes)}")
    
    if successful_modes > 0:
        print("\n📊 Results Files:")
        for result in results:
            if result['success']:
                print(f"   {result['description']}")
                print(f"   └── {result['output_file']}")
                print(f"       Duration: {result['duration']:.1f}s")
                print(f"       Completed Cases: {result['completed_cases']}")
            else:
                print(f"   {result['description']}: ❌ FAILED")
        
        print(f"\n🎉 All results saved in the 'results/' directory")
        print(f"   Each JSON file contains the evaluation mode in both filename and metadata")
        
        if successful_modes == len(modes):
            print(f"\n✨ All modes working! Ready to run full batch evaluation with:")
            print(f"   python run_batch_evaluation.py")
        
    else:
        print("\n❌ No evaluations completed successfully")
        sys.exit(1)

if __name__ == "__main__":
    main() 