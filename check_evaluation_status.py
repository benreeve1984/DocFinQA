#!/usr/bin/env python3
"""
Evaluation Status Checker for DocFinQA

This utility script checks the status of existing evaluation result files
and provides information about completion progress and resume capabilities.
"""

import os
import json
import glob
from datetime import datetime

def check_file_status(file_path):
    """
    Check the status of a single result file
    
    Args:
        file_path: Path to the result file
        
    Returns:
        Dictionary with file status information
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        metadata = data.get('metadata', {})
        results = data.get('results', [])
        
        # Extract key information
        status = {
            'file_path': file_path,
            'mode': metadata.get('evaluation_mode', 'unknown'),
            'model': metadata.get('model_used', 'unknown'),
            'completed_cases': len(results),
            'total_cases': metadata.get('total_cases', metadata.get('progress', {}).get('total_cases', 'unknown')),
            'timestamp': metadata.get('timestamp', 'unknown'),
            'success_rate': metadata.get('success_rate', 0),
            'numerical_accuracy': metadata.get('numerical_accuracy', 0),
            'is_complete': False,
            'query_rewriting': metadata.get('query_rewriting_enabled', False),
            'search_results': metadata.get('search_results_included', False)
        }
        
        # Check if complete
        if isinstance(status['total_cases'], int):
            status['is_complete'] = status['completed_cases'] >= status['total_cases']
        elif 'progress' in metadata:
            status['is_complete'] = metadata['progress'].get('is_complete', False)
        
        return status
        
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'mode': 'error',
            'completed_cases': 0,
            'is_complete': False
        }

def find_all_result_files():
    """
    Find all DocFinQA result files in the results directory
    
    Returns:
        List of file paths
    """
    if not os.path.exists('results'):
        return []
        
    pattern = "results/docfinqa_eval_*.json"
    return sorted(glob.glob(pattern), key=os.path.getctime, reverse=True)

def group_by_mode(file_statuses):
    """
    Group file statuses by evaluation mode
    
    Args:
        file_statuses: List of status dictionaries
        
    Returns:
        Dictionary grouped by mode
    """
    grouped = {}
    for status in file_statuses:
        mode = status['mode']
        if mode not in grouped:
            grouped[mode] = []
        grouped[mode].append(status)
    
    return grouped

def print_status_summary(file_statuses):
    """
    Print a summary of all evaluation statuses
    
    Args:
        file_statuses: List of status dictionaries
    """
    if not file_statuses:
        print("📋 No evaluation result files found in 'results/' directory")
        return
    
    print(f"📋 Found {len(file_statuses)} evaluation result files")
    print("=" * 60)
    
    # Group by mode
    grouped = group_by_mode(file_statuses)
    
    # Print summary by mode
    modes = ['closed_book', 'file_search_no_rewrite', 'file_search_with_rewrite', 'context_window']
    
    for mode in modes:
        if mode in grouped:
            mode_files = grouped[mode]
            print(f"\n🎯 {mode.replace('_', ' ').title()}")
            print("-" * 40)
            
            for status in mode_files:
                if 'error' in status:
                    print(f"   ❌ {os.path.basename(status['file_path'])}: ERROR - {status['error']}")
                    continue
                    
                # Status indicator
                if status['is_complete']:
                    indicator = "✅"
                elif status['completed_cases'] > 0:
                    indicator = "⚠️ "
                else:
                    indicator = "❌"
                
                # Format completion info
                if isinstance(status['total_cases'], int):
                    completion = f"{status['completed_cases']}/{status['total_cases']}"
                else:
                    completion = f"{status['completed_cases']} cases"
                
                print(f"   {indicator} {os.path.basename(status['file_path'])}")
                print(f"      Cases: {completion} | Accuracy: {status['numerical_accuracy']:.2%}")
                
                # Add special indicators
                indicators = []
                if status['query_rewriting']:
                    indicators.append("🔄 Query Rewriting")
                if status['search_results']:
                    indicators.append("🔍 Search Results")
                    
                if indicators:
                    print(f"      Features: {' | '.join(indicators)}")
    
    # Print other modes if any
    other_modes = set(grouped.keys()) - set(modes)
    for mode in other_modes:
        mode_files = grouped[mode]
        print(f"\n🔧 {mode}")
        print("-" * 40)
        for status in mode_files:
            print(f"   📄 {os.path.basename(status['file_path'])}: {status['completed_cases']} cases")

def print_resume_instructions(file_statuses):
    """
    Print instructions for resuming incomplete evaluations
    
    Args:
        file_statuses: List of status dictionaries
    """
    incomplete_files = [s for s in file_statuses if not s['is_complete'] and s['completed_cases'] > 0]
    
    if incomplete_files:
        print(f"\n🔄 RESUMPTION AVAILABLE")
        print("=" * 40)
        print(f"Found {len(incomplete_files)} incomplete evaluation(s) that can be resumed:")
        print()
        
        for status in incomplete_files:
            mode_desc = status['mode'].replace('_', ' ').title()
            completion = f"{status['completed_cases']}/{status['total_cases']}" if isinstance(status['total_cases'], int) else f"{status['completed_cases']} cases"
            print(f"   • {mode_desc}: {completion}")
        
        print(f"\n💡 To resume these evaluations:")
        print(f"   python run_batch_evaluation.py")
        print(f"   ")
        print(f"   The script will automatically detect and resume from existing files!")

def main():
    """Main entry point"""
    print("🔍 DocFinQA Evaluation Status Checker")
    print("=" * 50)
    
    # Find all result files
    result_files = find_all_result_files()
    
    if not result_files:
        print("📋 No evaluation result files found in 'results/' directory")
        print("\n💡 Run an evaluation first:")
        print("   python test_batch_evaluation.py  # for testing")
        print("   python run_batch_evaluation.py   # for full evaluation")
        return
    
    # Check status of each file
    file_statuses = []
    for file_path in result_files:
        status = check_file_status(file_path)
        file_statuses.append(status)
    
    # Print summary
    print_status_summary(file_statuses)
    
    # Print resume instructions if needed
    print_resume_instructions(file_statuses)
    
    print()

if __name__ == "__main__":
    main() 