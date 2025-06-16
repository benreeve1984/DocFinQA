# DocFinQA Batch Evaluation Scripts

This directory contains scripts for running comprehensive evaluations across all DocFinQA evaluation modes with **automatic resume capability** and **incremental progress saving**.

## Overview

The batch evaluation system tests 4 different evaluation modes:

1. **Closed Book** (`closed_book`) - No context provided to the model
2. **File Search (No Rewriting)** (`file_search_no_rewrite`) - Uses file_search tool without query rewriting  
3. **File Search (With Rewriting)** (`file_search_with_rewrite`) - Uses file_search tool with query rewriting
4. **Context Window** (`context_window`) - Full context provided directly in the prompt

## 🔄 **NEW: Resume Functionality**

**Never lose progress again!** The evaluation system now features:

- **Incremental Saving**: Results are saved after each individual test case
- **Automatic Resume**: If an evaluation fails or is interrupted, simply run the same command again
- **Smart Detection**: Automatically finds and resumes from existing partial result files
- **Progress Tracking**: Each JSON file includes completion progress information

## Scripts

### `run_batch_evaluation.py` - Full Batch Evaluation
Runs 100 test cases for each of the 4 evaluation modes with automatic resume.

**Usage:**
```bash
python run_batch_evaluation.py
```

**Features:**
- ✅ **Automatic Resume**: If interrupted, run again to continue from where it left off
- ✅ **Incremental Saving**: Progress saved after each test case
- ✅ **Smart File Detection**: Finds existing partial results from today
- ✅ **Progress Tracking**: Shows completion status for each mode

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

### `check_evaluation_status.py` - Status Checker
Check the status of existing evaluation results and get resume instructions.

**Usage:**
```bash
python check_evaluation_status.py
```

**Shows:**
- ✅ Completed evaluations
- ⚠️ Partial evaluations (with resume instructions)
- ❌ Failed evaluations
- 📊 Progress statistics and accuracy metrics

## JSON Output Format

Each result file contains:

### Filename
The evaluation mode is included in the filename for easy identification.

### Metadata with Progress Tracking
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
    "timestamp": "2025-06-16T22:00:00.000000",
    "progress": {
      "completed_cases": 75,
      "total_cases": 100,
      "completion_percentage": 75.0,
      "is_complete": false
    }
    // ... other metrics
  },
  "results": [
    // ... individual test case results
  ]
}
```

## Resume Functionality in Detail

### How It Works

1. **Incremental Saving**: After each test case, results are immediately saved to the JSON file
2. **Smart Detection**: When you run a batch evaluation, it checks for existing files from today with the same mode
3. **Question Matching**: Compares questions to determine which test cases are already completed
4. **Seamless Resume**: Continues from the next uncompleted test case

### Example Resume Flow

```bash
# Start evaluation
python run_batch_evaluation.py

# ... evaluation runs for 30 minutes, completes closed_book (100/100) 
#     and file_search_no_rewrite (45/100), then fails

# Check status
python check_evaluation_status.py
# Shows: closed_book ✅ complete, file_search_no_rewrite ⚠️ partial (45/100)

# Resume - automatically continues from case 46
python run_batch_evaluation.py
# Output: "Found existing partial results... Will resume from where it left off..."
```

### Interruption Handling

The system gracefully handles:

- **Network failures**: Partial results are saved
- **API rate limits**: Resumes from last completed case
- **Ctrl+C interruption**: Shows clear resume instructions
- **Script crashes**: All completed work is preserved
- **System restarts**: Resume by running the same command

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

### During Evaluation
```
📋 Mode 1/4: Closed Book (No Context)
------------------------------
🔄 Found existing partial results: results/docfinqa_eval_closed_book_20250616_180000.json
   Will resume from where it left off...
Starting fresh evaluation of 100 test cases
Found 23 already completed test cases
Resuming evaluation with 77 remaining cases
Evaluating test case 24/100 (remaining: 77)
...
```

### Completion Status
```
📊 Completion Status:
   ✅ Complete: 2/4 modes
   ⚠️  Partial: 1 modes
      - File Search (With Query Rewriting): 67/100 cases

💡 To complete remaining evaluations, simply run this script again!
```

## Example Output

```
🎯 DocFinQA Batch Evaluation
==================================================
Running 100 tests for each of 4 evaluation modes...

📋 Mode 1/4: Closed Book (No Context)
------------------------------
✅ closed_book completed successfully in 420.1s
   All 100 test cases completed

⏱️  Pausing 30 seconds before next mode...

📋 Mode 2/4: File Search (No Query Rewriting)
------------------------------
🔄 Found existing partial results: results/docfinqa_eval_file_search_no_rewrite_20250616_140000.json
   Will resume from where it left off...
✅ file_search_no_rewrite completed successfully in 1200.5s
   All 100 test cases completed

🏁 Batch Evaluation Complete!
==================================================

📊 Completion Status:
   ✅ Complete: 4/4 modes

🎊 All evaluations completed successfully!
```

## Troubleshooting

### If an evaluation fails:
1. **Check the status**: `python check_evaluation_status.py`
2. **Review the error message** in the terminal output
3. **Check API credits** and network connectivity
4. **Simply re-run** the same command - it will resume automatically

### If you want to start fresh:
1. **Delete existing result files** for the modes you want to restart
2. **Or rename the results directory** to preserve old results

### If resumption doesn't work:
1. **Check file permissions** in the results directory
2. **Verify JSON file integrity** - corrupted files will be skipped
3. **Run the test script first**: `python test_batch_evaluation.py`

## Advanced Usage

### Force Fresh Start
```bash
# Rename existing results to start completely fresh
mv results results_backup_$(date +%Y%m%d_%H%M%S)
python run_batch_evaluation.py
```

### Run Single Mode with Resume
```bash
# Run just one evaluation mode
python evals/docfinqa_eval.py --evaluation-mode closed_book --limit 100 --output results/my_closed_book_eval.json

# If it fails, run the exact same command again to resume
python evals/docfinqa_eval.py --evaluation-mode closed_book --limit 100 --output results/my_closed_book_eval.json
```

The batch script will continue with remaining modes even if one fails, and provides clear instructions for resuming incomplete work. 