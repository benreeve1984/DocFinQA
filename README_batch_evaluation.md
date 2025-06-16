# DocFinQA Batch Evaluation Scripts

This directory contains scripts for running comprehensive evaluations across all DocFinQA evaluation modes.

## Overview

The batch evaluation system tests 4 different evaluation modes:

1. **Closed Book** (`closed_book`) - No context provided to the model
2. **File Search (No Rewriting)** (`file_search_no_rewrite`) - Uses file_search tool without query rewriting  
3. **File Search (With Rewriting)** (`file_search_with_rewrite`) - Uses file_search tool with query rewriting
4. **Context Window** (`context_window`) - Full context provided directly in the prompt

## Scripts

### `run_batch_evaluation.py` - Full Batch Evaluation
Runs 100 test cases for each of the 4 evaluation modes.

**Usage:**
```bash
python run_batch_evaluation.py
```

**Expected Runtime:** ~2-3 hours (depending on API response times)

**Output:** 4 separate JSON files with descriptive names:
- `results/docfinqa_eval_closed_book_YYYYMMDD_HHMMSS.json`
- `results/docfinqa_eval_file_search_no_rewrite_YYYYMMDD_HHMMSS.json` 
- `results/docfinqa_eval_file_search_with_rewrite_YYYYMMDD_HHMMSS.json`
- `results/docfinqa_eval_context_window_YYYYMMDD_HHMMSS.json`

### `test_batch_evaluation.py` - Test Run
Runs 1 test case for each mode to verify everything works.

**Usage:**
```bash
python test_batch_evaluation.py
```

**Expected Runtime:** ~2 minutes

**Use this first** to verify your setup before running the full batch evaluation.

## JSON Output Format

Each result file contains:

### Filename
The evaluation mode is included in the filename for easy identification.

### Metadata
```json
{
  "metadata": {
    "evaluation_mode": "closed_book",
    "query_rewriting_enabled": false,
    "search_results_included": false,
    "model_used": "gpt-4.1-mini",
    "total_cases": 100,
    "successful_cases": 98,
    "numerical_accuracy": 0.85,
    "timestamp": "2025-06-16T22:00:00.000000"
    // ... other metrics
  },
  "results": [
    // ... individual test case results
  ]
}
```

## Prerequisites

1. **OpenAI API Key:** Set `OPENAI_API_KEY` environment variable
2. **Data File:** Ensure `data/test-data-sample.json` exists
3. **Dependencies:** Install required Python packages (see main README)

## Rate Limiting

The scripts include:
- 1 second pause between individual test cases
- 30 second pause between different evaluation modes
- Automatic retry logic for failed API calls

## Monitoring Progress

Both scripts provide clear progress indicators:
- Mode being run (1/4, 2/4, etc.)
- Individual test case progress
- Completion times for each mode
- Final summary with all result files

## Example Output

```
🎯 DocFinQA Batch Evaluation
==================================================
Running 100 tests for each of 4 evaluation modes...

📋 Mode 1/4: Closed Book (No Context)
------------------------------
🚀 Starting closed_book evaluation...
Evaluating test case 1...
Evaluating test case 2...
...
✅ closed_book completed successfully in 420.1s

⏱️  Pausing 30 seconds before next mode...

📋 Mode 2/4: File Search (No Query Rewriting)
------------------------------
...

🏁 Batch Evaluation Complete!
==================================================
Total time: 145.2 minutes

✅ Successful modes: 4/4

📊 Results Files:
   Closed Book (No Context)
   └── results/docfinqa_eval_closed_book_20250616_180000.json
```

## Troubleshooting

If a mode fails:
1. Check your OpenAI API key
2. Verify you have sufficient API credits
3. Check network connectivity
4. Run `test_batch_evaluation.py` to isolate the issue

The batch script will continue with remaining modes even if one fails. 