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

def run_evaluation(mode_name, mode_args, limit=1):
    """
    Run a single evaluation mode
    
    Args:
        mode_name: Descriptive name for the mode (used in filename)
        mode_args: List of command line arguments for the evaluation
        limit: Number of test cases to run
    
    Returns:
        (success, output_file, duration)
    """
    # Generate timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"results/docfinqa_eval_{mode_name}_{timestamp}.json"
    
    # Build the command
    cmd = [
        "python", "evals/docfinqa_eval.py",
        "--limit", str(limit),
        "--output", output_file
    ] + mode_args
    
    print(f"🚀 Starting {mode_name} evaluation...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Output: {output_file}")
    
    start_time = time.time()
    
    try:
        # Run the evaluation
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {mode_name} completed successfully in {duration:.1f}s")
            print(f"   Results saved to: {output_file}")
            return True, output_file, duration
        else:
            print(f"❌ {mode_name} failed!")
            print(f"   Error: {result.stderr}")
            return False, None, duration
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ {mode_name} failed with exception: {e}")
        return False, None, duration

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
        
        success, output_file, duration = run_evaluation(
            mode_name=mode['name'],
            mode_args=mode['args'],
            limit=1  # Just 1 test case for testing
        )
        
        results.append({
            'mode': mode['name'],
            'description': mode['description'],
            'success': success,
            'output_file': output_file,
            'duration': duration
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