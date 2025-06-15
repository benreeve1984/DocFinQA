#!/usr/bin/env python3
"""
DocFinQA Evaluation Script using OpenAI Responses API (Agents SDK)

=== WHAT THIS SCRIPT DOES ===
This script evaluates how well OpenAI's AI models can answer financial questions
by reading through financial documents using the new Responses API with file_search tool.

=== HOW IT WORKS ===
1. Takes a JSON file with financial documents and questions about them
2. For each question:
   - Uploads the financial document to OpenAI's vector store (searchable database)
   - Uses the Responses API with file_search tool to answer the question
   - Compares the answer to the correct answer using numerical scoring
   - Cleans up (deletes the temporary document storage)
3. Calculates overall performance statistics
4. Saves detailed results to a JSON file

=== KEY CONCEPTS FOR BEGINNERS ===
- Vector Store: A searchable database where OpenAI stores your documents
- Responses API: The new stateful API that can use tools like file_search
- file_search: A tool that lets the AI search through uploaded documents
- Agents SDK: The modern way to build AI applications with OpenAI

=== USAGE EXAMPLES ===
Basic usage (all test cases):
    python docfinqa_eval.py

Test with just 5 cases:
    python docfinqa_eval.py --limit 5 --output test_results.json

Use different model:
    python docfinqa_eval.py --model gpt-4o-mini --limit 10
"""

# === PYTHON STANDARD LIBRARY IMPORTS ===
import json          # For reading/writing JSON files (test data, results)
import os            # For environment variables and file system operations
import sys           # For system-specific parameters and functions
import time          # For timing operations and adding delays
import argparse      # For command-line argument parsing
import logging       # For detailed logging and debugging
from typing import Dict, List, Any, Optional  # For type hints (better code documentation)
from dataclasses import dataclass             # For creating simple data structures
from datetime import datetime                 # For timestamps in results
import tempfile      # For creating temporary files (document uploads)

# === OPENAI SDK IMPORTS ===
# These are the official OpenAI Python library components we need
from openai import OpenAI                          # Main OpenAI client

# === ENVIRONMENT SETUP ===
# Load environment variables from .env file (like API keys)
from dotenv import load_dotenv
load_dotenv()  # This reads your .env file and makes variables available

# === IMPORT OUR CUSTOM SCORER ===
# This is our custom scoring system for evaluating financial Q&A responses
import sys
import os
# Add the current directory to Python's path so we can import our scorer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scorer import score_response, calculate_aggregate_scores

# === LOGGING SETUP ===
# Configure logging so we can see what's happening during evaluation
logging.basicConfig(
    level=getattr(logging, os.getenv('OPENAI_LOG_LEVEL', 'INFO')),  # Log level from env var
    format='%(asctime)s - %(levelname)s - %(message)s'              # Log format with timestamp
)
logger = logging.getLogger(__name__)  # Create logger for this script

# === DATA STRUCTURES ===
# These are simple data containers to organize our information

@dataclass
class TestCase:
    """
    Represents a single test case from the dataset
    
    Think of this as one row from your test data - it contains:
    - A financial document (context)
    - A question about that document  
    - The correct answer to that question
    - An optional program/reasoning (usually empty)
    
    Example:
        context = "Company XYZ reported Q3 revenue of $100M..."
        question = "What was Q3 revenue?"
        expected_answer = "100"
    """
    context: str         # The financial document text (can be very long!)
    question: str        # The question to ask about the document
    expected_answer: str # The correct answer we're comparing against
    program: str = ""    # Optional reasoning steps (usually not used)
    
@dataclass
class EvalResult:
    """
    Represents the result of evaluating a single test case
    
    This contains everything that happened when we tested one question:
    - The original test case
    - What the AI responded
    - How long it took
    - Technical details (IDs for cleanup)
    - Whether it succeeded
    - Scoring information
    
    Example:
        If we asked "What was Q3 revenue?" and expected "100":
        - model_response might be "FINAL: 100.0"
        - numerical_accuracy would be 1.0 (perfect match)
        - extracted_number would be 100.0
        - success would be True
    """
    test_case: TestCase           # The original test case we evaluated
    model_response: str           # What the AI actually said
    execution_time: float         # How many seconds the evaluation took
    vector_store_id: str          # OpenAI vector store ID (for cleanup)
    response_id: str              # OpenAI response ID (for reference)
    success: bool                 # True if evaluation completed, False if error
    error_message: Optional[str] = None  # Error message if something went wrong
    
    # === SCORING INFORMATION ===
    # These fields contain the results of our numerical accuracy scoring
    numerical_accuracy: Optional[float] = None  # 0.0 to 1.0, our main metric
    extracted_number: Optional[float] = None    # Number we found in AI response
    expected_number: Optional[float] = None     # Number we expected to find
    found_final_tag: Optional[bool] = None      # Did AI use "FINAL: X" format?
    
# === MAIN EVALUATOR CLASS ===
# This is the heart of our evaluation system

class DocFinQAEvaluator:
    """
    Main evaluator class for DocFinQA using OpenAI Responses API (Agents SDK)
    
    This class handles the entire evaluation process:
    1. Connects to OpenAI API
    2. Manages document uploads to vector stores
    3. Uses Responses API with file_search tool to answer questions
    4. Calculates scores and metrics
    5. Cleans up resources
    
    Think of this as your "evaluation manager" - it coordinates everything.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4.1-mini",
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize the evaluator - sets up connection to OpenAI and configuration
        
        Args:
            api_key: Your OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: Which AI model to use (gpt-4o, gpt-4o-mini, etc.)
            max_retries: How many times to retry if API calls fail
            retry_delay: How long to wait between retries (in seconds)
            
        Example:
            evaluator = DocFinQAEvaluator(model="gpt-4o-mini", max_retries=5)
        """
        # Create OpenAI client - this is our connection to OpenAI's servers
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
        # Store configuration settings
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')  # AI model to use
        self.max_retries = max_retries      # How many times to retry failed calls
        self.retry_delay = retry_delay      # Seconds to wait between retries
        
        logger.info(f"Initialized DocFinQA Evaluator with model: {self.model}")
        
    def load_test_data(self, file_path: str, limit: Optional[int] = None) -> List[TestCase]:
        """
        Load test cases from JSON file
        
        Args:
            file_path: Path to the JSON file containing test data
            limit: Optional limit on number of test cases to load
            
        Returns:
            List of TestCase objects
            
        Example:
            test_cases = evaluator.load_test_data("data/test-data-sample.json", limit=10)
        """
        logger.info(f"Loading test data from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            test_cases = []
            for item in data[:limit] if limit else data:
                test_case = TestCase(
                    context=item.get('Context', ''),
                    question=item.get('Question', ''),
                    expected_answer=str(item.get('Answer', '')),
                    program=item.get('Program', '')
                )
                test_cases.append(test_case)
                
            logger.info(f"Loaded {len(test_cases)} test cases")
            return test_cases
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise
            
    def create_vector_store_with_context(self, context: str, name: str = None) -> str:
        """
        Create a vector store and upload the context document
        
        Args:
            context: The document context to upload
            name: Optional name for the vector store
            
        Returns:
            Vector store ID string
        """
        # Create a temporary file with the context
        # Use .json extension as it's reliably supported by OpenAI
        # We'll format the context as a simple JSON document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            # Format context as JSON with proper structure
            import json
            content_data = {
                "document_type": "financial_context",
                "content": context
            }
            json.dump(content_data, f, ensure_ascii=False, indent=2)
            temp_file_path = f.name
            
        try:
            # Create vector store - try both beta and non-beta paths for compatibility
            vector_store_name = name or f"docfinqa-{int(time.time())}"
            try:
                # Try new path first (v1.66.x+)
                vector_store = self.client.vector_stores.create(
                    name=vector_store_name,
                    expires_after={"anchor": "last_active_at", "days": 1}  # Auto-delete after 1 day
                )
                vector_stores_api = self.client.vector_stores
            except AttributeError:
                # Fall back to beta path for older versions
                vector_store = self.client.beta.vector_stores.create(
                    name=vector_store_name,
                    expires_after={"anchor": "last_active_at", "days": 1}  # Auto-delete after 1 day
                )
                vector_stores_api = self.client.beta.vector_stores
            
            # Upload file to vector store
            with open(temp_file_path, 'rb') as f:
                file_response = self.client.files.create(
                    file=f,
                    purpose='assistants'
                )
                
            # Add file to vector store
            vector_stores_api.files.create(
                vector_store_id=vector_store.id,
                file_id=file_response.id
            )
            
            # Wait for file to be processed
            max_wait = 120  # Maximum wait time in seconds (increased for large documents)
            wait_time = 0
            check_interval = 2  # Check every 2 seconds instead of every 1 second
            
            while wait_time < max_wait:
                vector_store_files = vector_stores_api.files.list(
                    vector_store_id=vector_store.id
                )
                
                if vector_store_files.data and vector_store_files.data[0].status == 'completed':
                    logger.debug(f"Vector store file processing completed in {wait_time}s")
                    break
                elif vector_store_files.data and vector_store_files.data[0].status == 'failed':
                    raise RuntimeError(f"Vector store file processing failed: {vector_store_files.data[0].last_error}")
                    
                time.sleep(check_interval)
                wait_time += check_interval
                
                # Log progress every 30 seconds
                if wait_time % 30 == 0:
                    logger.info(f"Still processing vector store file... ({wait_time}s elapsed)")
                
            if wait_time >= max_wait:
                raise TimeoutError(f"Vector store file processing timed out after {max_wait} seconds")
                
            logger.debug(f"Created vector store {vector_store.id} with context document")
            return vector_store.id
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    def evaluate_single_case(self, test_case: TestCase, case_index: int) -> EvalResult:
        """
        Evaluate a single test case using the Responses API
        
        This method takes one financial question and document, then:
        1. Uploads the document to OpenAI as a searchable vector store
        2. Uses the Responses API with file_search tool to answer the question
        3. Scores the response for accuracy
        4. Cleans up all the temporary resources
        
        Args:
            test_case: The test case to evaluate (document + question + expected answer)
            case_index: Index of the test case (for logging, starts from 0)
            
        Returns:
            EvalResult object containing everything that happened
            
        Example:
            result = evaluator.evaluate_single_case(test_cases[0], 0)
            print(f"AI said: {result.model_response}")
            print(f"Accuracy: {result.numerical_accuracy}")
        """
        # Record when we started (for timing how long this takes)
        start_time = time.time()
        
        # Initialize tracking variables for OpenAI resources
        vector_store_id = None  # Where we store the document
        response_id = None      # The Responses API response ID
        
        try:
            logger.info(f"Evaluating test case {case_index + 1}")
            
            # === STEP 1: UPLOAD DOCUMENT ===
            # Create a vector store (searchable database) with the financial document
            vector_store_id = self.create_vector_store_with_context(
                test_case.context,                    # The financial document text
                f"docfinqa-case-{case_index}"        # Give it a unique name
            )
            
            # === STEP 2: USE RESPONSES API WITH FILE_SEARCH ===
            # Create a formatted prompt with the question and instructions
            prompt = f"""### Question
{test_case.question}

### Instructions
Use the file_search tool to find information from the uploaded financial document to answer the question.

Work step by step under the header REASONING: and explain your thinking.
Finish with a single line with only your numerical or logical answer under the header FINAL: <answer>
"""
            
            # Use the Responses API with file_search tool
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id]
                }]
            )
            
            response_id = response.id
            
            # Extract the AI's response text
            model_response = ""
            for output in response.output:
                if hasattr(output, 'content') and output.content:
                    for content in output.content:
                        if hasattr(content, 'text'):
                            model_response += content.text + "\n"
            
            model_response = model_response.strip()
            
            # === STEP 3: SCORE THE RESPONSE ===
            # Use our custom scoring system to evaluate the response
            scoring_result = score_response(model_response, test_case.expected_answer)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            logger.info(f"Successfully evaluated test case {case_index + 1} in {execution_time:.2f}s")
            
            # Return successful result with scoring information
            return EvalResult(
                test_case=test_case,
                model_response=model_response,
                execution_time=execution_time,
                vector_store_id=vector_store_id,
                response_id=response_id,
                success=True,
                # Scoring information from our custom scorer (ScoreResult object)
                numerical_accuracy=scoring_result.numerical_accuracy,
                extracted_number=scoring_result.extracted_number,
                expected_number=scoring_result.expected_number,
                found_final_tag=scoring_result.found_final_tag
            )
            
        except Exception as e:
            # Something went wrong - log the error and return a failed result
            logger.error(f"Error evaluating test case {case_index + 1}: {e}")
            execution_time = time.time() - start_time
            
            return EvalResult(
                test_case=test_case,
                model_response="",                           # No response due to error
                execution_time=execution_time,
                vector_store_id=vector_store_id or "",       # Might be None if creation failed
                response_id=response_id or "",               # Might be None if API call failed
                success=False,                               # Mark as failed
                error_message=str(e)                         # What went wrong
            )
            
        finally:
            # === CLEANUP ===  
            # This runs whether we succeeded or failed
            # Always clean up OpenAI resources to avoid charges
            self.cleanup_resources(vector_store_id)
            
    def cleanup_resources(self, vector_store_id: Optional[str]):
        """
        Clean up OpenAI resources
        
        Args:
            vector_store_id: ID of vector store to delete
        """
        # Clean up vector store - try both API paths for compatibility
        try:
            if vector_store_id:
                try:
                    # Try new path first (v1.66.x+)
                    self.client.vector_stores.delete(vector_store_id)
                    logger.debug(f"Deleted vector store {vector_store_id}")
                except AttributeError:
                    # Fall back to beta path for older versions
                    self.client.beta.vector_stores.delete(vector_store_id)
                    logger.debug(f"Deleted vector store {vector_store_id}")
        except Exception as e:
            logger.warning(f"Failed to delete vector store {vector_store_id}: {e}")
            
    def evaluate_all(self, test_cases: List[TestCase]) -> List[EvalResult]:
        """
        Evaluate all test cases
        
        Args:
            test_cases: List of test cases to evaluate
            
        Returns:
            List of EvalResult objects
        """
        results = []
        
        logger.info(f"Starting evaluation of {len(test_cases)} test cases")
        
        for i, test_case in enumerate(test_cases):
            try:
                result = self.evaluate_single_case(test_case, i)
                results.append(result)
                
                # Brief pause between evaluations to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Fatal error on test case {i + 1}: {e}")
                # Continue with next test case
                continue
                
        return results
        
    def calculate_metrics(self, results: List[EvalResult]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary of metrics
        """
        total_cases = len(results)
        successful_cases = sum(1 for r in results if r.success)
        failed_cases = total_cases - successful_cases
        
        if successful_cases > 0:
            avg_execution_time = sum(r.execution_time for r in results if r.success) / successful_cases
        else:
            avg_execution_time = 0
            
        # Calculate numerical accuracy metrics for successful cases
        successful_results = [r for r in results if r.success and r.numerical_accuracy is not None]
        
        if successful_results:
            avg_numerical_accuracy = sum(r.numerical_accuracy for r in successful_results) / len(successful_results)
            perfect_scores = sum(1 for r in successful_results if r.numerical_accuracy == 1.0)
            extraction_success = sum(1 for r in successful_results if r.extracted_number is not None)
            final_tag_usage = sum(1 for r in successful_results if r.found_final_tag)
        else:
            avg_numerical_accuracy = 0
            perfect_scores = 0
            extraction_success = 0
            final_tag_usage = 0
            
        metrics = {
            # Basic execution metrics
            'total_cases': total_cases,
            'successful_cases': successful_cases,
            'failed_cases': failed_cases,
            'success_rate': successful_cases / total_cases if total_cases > 0 else 0,
            'average_execution_time': avg_execution_time,
            
            # Numerical accuracy metrics (primary for financial Q&A)
            'numerical_accuracy': avg_numerical_accuracy,
            'perfect_answers': perfect_scores,
            'perfect_answer_rate': perfect_scores / len(successful_results) if successful_results else 0,
            
            # Extraction statistics
            'extraction_success_count': extraction_success,
            'extraction_success_rate': extraction_success / len(successful_results) if successful_results else 0,
            'final_tag_usage_count': final_tag_usage,
            'final_tag_usage_rate': final_tag_usage / len(successful_results) if successful_results else 0,
            
            # Metadata
            'model_used': self.model,
            'timestamp': datetime.now().isoformat(),
            'scored_cases': len(successful_results)
        }
        
        return metrics
        
    def save_results(self, results: List[EvalResult], metrics: Dict[str, Any], output_file: str):
        """
        Save evaluation results to JSON file
        
        Args:
            results: List of evaluation results
            metrics: Evaluation metrics
            output_file: Path to output file
        """
        output_data = {
            'metadata': metrics,
            'results': []
        }
        
        for result in results:
            result_data = {
                'question': result.test_case.question,
                'expected_answer': result.test_case.expected_answer,
                'model_response': result.model_response,
                'execution_time': result.execution_time,
                'success': result.success,
                'error_message': result.error_message,
                'vector_store_id': result.vector_store_id,
                'response_id': result.response_id,
                # Scoring information
                'numerical_accuracy': result.numerical_accuracy,
                'extracted_number': result.extracted_number,
                'expected_number': result.expected_number,
                'found_final_tag': result.found_final_tag
            }
            output_data['results'].append(result_data)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Results saved to {output_file}")

def main():
    """Main entry point for the evaluation script"""
    parser = argparse.ArgumentParser(description='DocFinQA Evaluation using OpenAI Responses API (Agents SDK)')
    parser.add_argument('--data', '-d', 
                       default='data/test-data-sample.json',
                       help='Path to test data JSON file')
    parser.add_argument('--output', '-o',
                       default='eval_results.json',
                       help='Path to output results file')
    parser.add_argument('--limit', '-l',
                       type=int,
                       help='Limit number of test cases to process')
    parser.add_argument('--model', '-m',
                       default=None,
                       help='OpenAI model to use (overrides env var)')
    
    args = parser.parse_args()
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)
        
    try:
        # Initialize evaluator
        evaluator = DocFinQAEvaluator(model=args.model)
        
        # Load test data
        test_cases = evaluator.load_test_data(args.data, args.limit)
        
        if not test_cases:
            logger.error("No test cases loaded")
            sys.exit(1)
            
        # Run evaluation
        results = evaluator.evaluate_all(test_cases)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(results)
        
        # Save results
        evaluator.save_results(results, metrics, args.output)
        
        # Print summary
        print("\nEvaluation Complete!")
        print(f"Total cases: {metrics['total_cases']}")
        print(f"Successful: {metrics['successful_cases']}")
        print(f"Failed: {metrics['failed_cases']}")
        print(f"Success rate: {metrics['success_rate']:.2%}")
        print(f"Average execution time: {metrics['average_execution_time']:.2f}s")
        print(f"Numerical accuracy: {metrics['numerical_accuracy']:.2%}")
        print(f"Perfect answers: {metrics['perfect_answers']}/{metrics['scored_cases']}")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 