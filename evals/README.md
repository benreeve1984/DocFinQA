# DocFinQA Evaluation Suite

This directory contains evaluation scripts for testing financial document question-answering capabilities using OpenAI's Agents SDK with file_search.

## Overview

The evaluation suite processes financial documents and questions to test the accuracy of AI-powered document analysis. It uses OpenAI's Assistants API with vector stores and file_search functionality.

## Files

- `docfinqa_eval.py` - Main evaluation script
- `scorer.py` - Scoring and metrics calculation module
- `README.md` - This documentation file

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the `.env-template` to `.env` and fill in your OpenAI API credentials:

```bash
cp .env-template .env
```

Edit `.env` and set your OpenAI API key:

```
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Prepare Test Data

Ensure your test data file is in the correct format. Expected JSON structure:

```json
[
  {
    "Context": "Large financial document text...",
    "Question": "What was the revenue growth?",
    "Answer": "15%",
    "Program": ""
  }
]
```

## Usage

### Basic Evaluation

Run evaluation on all test cases:

```bash
python evals/docfinqa_eval.py
```

### Limited Evaluation

Run evaluation on first 5 test cases:

```bash
python evals/docfinqa_eval.py --limit 5
```

### Custom Configuration

```bash
python evals/docfinqa_eval.py \
  --data data/test-data-sample.json \
  --output my_results.json \
  --model gpt-4o \
  --limit 10
```

### Custom System Prompt

Create a custom prompt file and use it:

```bash
echo "You are an expert financial analyst..." > custom_prompt.txt
python evals/docfinqa_eval.py --prompt custom_prompt.txt
```

## Command Line Options

- `--data, -d`: Path to test data JSON file (default: `data/test-data-sample.json`)
- `--output, -o`: Path to output results file (default: `eval_results.json`)
- `--limit, -l`: Limit number of test cases to process
- `--model, -m`: OpenAI model to use (overrides env var)
- `--prompt, -p`: Path to custom system prompt file

## Output Format

The evaluation produces a JSON file with the following structure:

```json
{
  "metadata": {
    "total_cases": 100,
    "successful_cases": 95,
    "failed_cases": 5,
    "success_rate": 0.95,
    "average_execution_time": 12.5,
    "model_used": "gpt-4o",
    "timestamp": "2024-01-01T00:00:00"
  },
  "results": [
    {
      "question": "What was the revenue?",
      "expected_answer": "$100M",
      "model_response": "The revenue was $100 million.",
      "execution_time": 10.2,
      "success": true,
      "error_message": null,
      "vector_store_id": "vs_123",
      "assistant_id": "asst_456",
      "thread_id": "thread_789",
      "run_id": "run_abc"
    }
  ]
}
```

## Prompt Engineering

The evaluation uses a carefully crafted prompt adapted from your original financial Q&A prompt, optimized for the OpenAI Agents SDK with file_search:

### System Prompt Structure
- **Financial Q&A Agent**: Specialized role for financial document analysis
- **GLOBAL RULES**: Strict adherence to document-only information
- **Number Handling**: Exact copying of numbers with original formatting
- **Structured Output**: REASONING section followed by FINAL: answer

### User Message Format
```
### Question
{question}

### Context Instructions
Use the file_search tool to find information from the uploaded financial document.

### Footer Instructions (repeat key rules)
• Use only information from the uploaded document accessible via file_search.
• Work step by step under the header REASONING:.
• Finish with a single line: FINAL: <answer>

### Repeat Question (long-context anchor)
{question}
```

### Expected Response Format
```
REASONING:
• From the document: "Q3 2023 revenue was $150.5 million"
• The exact number is 150.5

FINAL: 150.5
```

### Key Adaptations from Original Prompt
- **Removed `<doc>` tags**: Now uses file_search instead of embedded context
- **Updated instructions**: References file_search tool instead of doc tags
- **Maintained structure**: Kept REASONING + FINAL format for consistency
- **Preserved tolerances**: Your numerical scorer handles the same ±0.5/±2% tolerances

## Scoring Module

The `scorer.py` module provides numerical accuracy scoring optimized for financial Q&A:

### Primary Metric
- **Numerical Accuracy**: Extracts numbers from "FINAL: X.XX" or fallback to last number
- **Financial Tolerances**: ±0.5 absolute or ±2% relative tolerance for rounding
- **Smart Extraction**: Handles various formats (currencies, percentages, etc.)

### Additional Metrics (for analysis)
- **Exact Match**: Exact string comparison
- **Normalized Exact Match**: Case-insensitive, punctuation-normalized comparison  
- **F1 Score**: Token-level F1 score
- **BLEU Score**: BLEU-like similarity score
- **Semantic Similarity**: String similarity using sequence matching

### Why Numerical Accuracy is Primary
- **Domain-Specific**: Perfect for financial Q&A where answers are typically numbers
- **Robust**: Handles both "FINAL: 21.48" and "The answer is 21.48" formats
- **Appropriate Tolerances**: Allows for reasonable rounding differences in financial calculations
- **Binary Success**: Clear pass/fail metrics ideal for evaluation

### Using the Scorer

```python
from scorer import score_response, calculate_aggregate_scores

# Score a single financial Q&A response
score = score_response(
    predicted="FINAL: 21.48",
    expected="21.48",
    question="What was the Q3 revenue in millions?"
)

# Primary metric for financial Q&A
print(f"Numerical Accuracy: {score.numerical_accuracy}")  # 1.0 (perfect match)
print(f"Extracted Number: {score.extracted_number}")      # 21.48
print(f"Expected Number: {score.expected_number}")        # 21.48
print(f"Found FINAL Tag: {score.found_final_tag}")        # True

# Additional metrics
print(f"F1 Score: {score.f1_score}")
print(f"Exact Match: {score.exact_match}")

# Calculate aggregate scores for evaluation run
scores = [score1, score2, score3]  # List of ScoreResult objects
aggregates = calculate_aggregate_scores(scores)
print(f"Average Numerical Accuracy: {aggregates['numerical_accuracy']}")
print(f"Perfect Answer Rate: {aggregates['perfect_answer_rate']}")
print(f"Extraction Success Rate: {aggregates['extraction_success_rate']}")
```

## Architecture

### Evaluation Flow

1. **Load Test Data**: Parse JSON file with financial documents and questions
2. **For Each Test Case**:
   - Create temporary file with document context
   - Upload to OpenAI vector store
   - Create assistant with file_search tool
   - Create conversation thread
   - Send question to assistant
   - Wait for response
   - Collect response and metadata
   - Clean up resources (vector store, assistant)
3. **Calculate Metrics**: Score responses against expected answers
4. **Save Results**: Output comprehensive evaluation report

### Resource Management

The evaluation script automatically:
- Creates temporary files for document upload
- Creates and destroys vector stores for each test case
- Creates and destroys assistants for each test case
- Implements timeouts to prevent hanging
- Provides detailed logging and error handling

## Configuration Options

### Environment Variables

All configuration can be done via environment variables (see `.env-template`):

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: gpt-4o)
- `ASSISTANT_RUN_TIMEOUT`: Max time to wait for responses (default: 300s)
- `FILE_PROCESSING_TIMEOUT`: Max time for file processing (default: 30s)
- `OPENAI_LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

### System Prompt Customization

The default system prompt can be overridden by:

1. Using the `--prompt` command line argument
2. Calling `evaluator.set_system_prompt()` in code
3. Setting a custom prompt in the DocFinQAEvaluator constructor

Example custom prompt:

```
You are a financial document analysis expert with 20 years of experience. 
When analyzing documents, focus on:
1. Numerical accuracy and precision
2. Proper citation of sources
3. Clear, concise answers
4. Acknowledgment of limitations or uncertainty

Always cite specific sections or page numbers when possible.
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `OPENAI_API_KEY` is set in your `.env` file
2. **Rate Limits**: The script includes delays between requests. Adjust `DELAY_BETWEEN_REQUESTS` if needed
3. **Timeouts**: Large documents may take time to process. Increase timeout values if needed
4. **Memory Issues**: Large context documents may cause memory issues. Consider chunking for very large files

### Debugging

Enable debug logging:

```bash
export OPENAI_LOG_LEVEL=DEBUG
python evals/docfinqa_eval.py --limit 1
```

This will show detailed API calls, timing information, and intermediate results.

### Error Recovery

The script is designed to be robust:
- Individual test case failures don't stop the entire evaluation
- Resources are cleaned up even if errors occur
- Detailed error messages are logged and saved in results
- Partial results are saved if the evaluation is interrupted

## Customization

### Adding Custom Scoring

To add domain-specific scoring, modify the `calculate_custom_score` function in `scorer.py`:

```python
def calculate_custom_score(pred_text: str, gold_text: str, question: str = "") -> float:
    score = 0.0
    
    # Check for financial terms
    financial_terms = ['revenue', 'profit', 'loss', 'earnings']
    for term in financial_terms:
        if term in pred_text.lower() and term in gold_text.lower():
            score += 0.1
    
    # Check for numerical accuracy
    pred_numbers = extract_numbers(pred_text)
    gold_numbers = extract_numbers(gold_text)
    if pred_numbers == gold_numbers:
        score += 0.5
    
    return min(score, 1.0)
```

### Extending the Evaluator

To add new functionality, inherit from `DocFinQAEvaluator`:

```python
class CustomEvaluator(DocFinQAEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_config = load_custom_config()
    
    def evaluate_single_case(self, test_case, case_index):
        # Add custom preprocessing
        test_case = self.preprocess_case(test_case)
        
        # Call parent method
        result = super().evaluate_single_case(test_case, case_index)
        
        # Add custom postprocessing
        result = self.postprocess_result(result)
        
        return result
```

## Performance Considerations

- **Parallel Processing**: Currently processes one test case at a time. Consider implementing parallel processing for faster evaluation
- **Vector Store Reuse**: For similar documents, vector stores could potentially be reused
- **Caching**: Implement response caching for repeated questions
- **Batch Processing**: Group similar documents for more efficient processing

## Contributing

When contributing to this evaluation suite:

1. Add tests for new functionality
2. Update documentation for new features
3. Follow the existing code style and patterns
4. Add appropriate logging and error handling
5. Consider backward compatibility

## License

This evaluation suite is part of the DocFinQA project. 