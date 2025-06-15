# Beginner's Guide to the DocFinQA Evaluation Code

## 📚 Overview

This guide explains the DocFinQA evaluation system in simple terms. The code is now heavily annotated to help beginners understand every step.

## 🗂️ Code Structure

### Main Files
- `evals/docfinqa_eval.py` - Main evaluation script (heavily annotated)
- `evals/scorer.py` - Scoring system (heavily annotated)  
- `data/test-data-sample.json` - Test data with financial documents and questions
- `.env` - Your OpenAI API key and configuration

## 🔍 Key Concepts Explained

### What is DocFinQA?
DocFinQA evaluates how well AI can answer financial questions by reading financial documents (like earnings reports, SEC filings, etc.).

### How the Evaluation Works
1. **Load Test Data**: Read questions and financial documents from JSON file
2. **For Each Question**:
   - Upload the financial document to OpenAI (creates a "vector store")
   - Create an AI assistant that can search through the document
   - Ask the assistant the financial question
   - Get the assistant's answer
   - Score the answer for numerical accuracy
   - Clean up (delete the temporary assistant and document)
3. **Calculate Statistics**: Overall accuracy, success rates, etc.

### Key OpenAI Concepts
- **Vector Store**: A searchable database where OpenAI stores your documents
- **Assistant**: An AI agent that can use tools (like file_search) to answer questions
- **Thread**: A conversation between you and an assistant
- **Run**: A single execution of the assistant processing your message
- **file_search**: A tool that lets the assistant search through uploaded documents

## 📊 Scoring System

### Why Numerical Accuracy?
Financial Q&A is about getting the right numbers. Traditional text similarity metrics don't work well when comparing "21.48" vs "21.50" - we need domain-specific scoring.

### How Numerical Scoring Works
1. **Extract Number from AI Response**: Look for "FINAL: 123.45" format first, then any number
2. **Extract Number from Expected Answer**: Same logic
3. **Compare with Tolerance**: Allow ±0.5 absolute or ±2% relative difference for rounding
4. **Return Score**: 1.0 if match, 0.0 if no match

### Examples
- AI says "FINAL: 21.48", expected "21.48" → Score: 1.0 (perfect)
- AI says "FINAL: 21.47", expected "21.48" → Score: 1.0 (within tolerance)  
- AI says "FINAL: 25.0", expected "21.48" → Score: 0.0 (too different)
- AI says "I don't know", expected "21.48" → Score: 0.0 (no number found)

## 🔧 Code Annotations Added

### Main Evaluation Script (`docfinqa_eval.py`)
- **Detailed imports explanation**: What each library does
- **Data structures explained**: What TestCase and EvalResult contain
- **Step-by-step evaluation process**: 9 clear steps from document upload to scoring
- **Error handling**: What happens when things go wrong
- **Resource cleanup**: Why we delete assistants and vector stores

### Scorer Module (`scorer.py`)
- **Number extraction logic**: How we find numbers in AI responses
- **Tolerance explanation**: Why we allow small differences
- **Fallback logic**: What happens when AI doesn't use "FINAL:" format
- **Examples throughout**: Real examples of inputs and outputs

## 🚀 Running the Code

### Basic Usage
```bash
# Test with 5 cases first
python evals/docfinqa_eval.py --limit 1 --output test_results.json

# Check the results
cat test_results.json | jq '.metadata.numerical_accuracy'
```

### Understanding Results
The output includes:
- `numerical_accuracy`: Main metric (0.0 to 1.0)
- `perfect_answer_rate`: % of exactly correct answers
- `extraction_success_rate`: % where we found a number in AI response
- `final_tag_usage_rate`: % where AI used "FINAL: X" format

## 🎯 Key Success Factors

1. **Proper Prompt**: The AI gets clear instructions to use "FINAL: X" format
2. **Document Quality**: Financial documents should be clear and searchable
3. **Numerical Tolerance**: Small rounding differences are allowed
4. **Error Handling**: Individual failures don't crash the entire evaluation
5. **Resource Management**: Always clean up OpenAI resources to avoid charges

## 🔍 Debugging Tips

1. **Enable Debug Logging**: `export OPENAI_LOG_LEVEL=DEBUG`
2. **Start Small**: Use `--limit 1` to test single cases
3. **Check API Key**: Make sure `OPENAI_API_KEY` is set in `.env`
4. **Monitor Costs**: Each evaluation uses OpenAI API calls
5. **Review Responses**: Look at `model_response` field in results

## 📖 Learning Path

1. **Start Here**: Read this guide and the code comments
2. **Run Simple Test**: `python evals/docfinqa_eval.py --limit 1`
3. **Check Results**: Look at the JSON output to understand scoring
4. **Modify Prompt**: Try different instructions to the AI
5. **Scale Up**: Run larger evaluations once you understand the basics

The code is now designed to be self-explanatory with extensive comments. Each major function and logic block includes beginner-friendly explanations! 