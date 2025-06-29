#!/usr/bin/env python3
"""
DocFinQA Evaluation Script using OpenAI Responses API

=== WHAT THIS SCRIPT DOES ===
This script evaluates how well OpenAI's AI models can answer financial questions
using three different evaluation modes that test different capabilities.

=== EVALUATION MODES ===
1. **file_search**: Uses OpenAI's file_search tool with vector stores (default)
   - Uploads financial documents to OpenAI's searchable database
   - Uses file_search tool to find relevant information
   - Tests the model's ability to search and retrieve information
   - Supports query rewriting and search result inclusion
   
2. **closed_book**: Tests model's base financial knowledge without context
   - No financial documents provided to the model
   - Tests what the model knows from its training data
   - Useful for understanding baseline capabilities
   
3. **context_window**: Provides full context directly in the prompt
   - Includes the entire financial document in the prompt
   - Tests the model's reasoning with complete information
   - Limited by model's context window size

=== HOW IT WORKS ===
1. Takes a JSON file with financial documents and questions about them
2. For each question, uses the specified evaluation mode to get an answer
3. Compares the answer to the correct answer using numerical scoring
4. Calculates overall performance statistics
5. Saves detailed results to a JSON file

=== USAGE EXAMPLES ===

Basic file_search mode (default):
    python docfinqa_eval.py

Test closed book mode (no context):
    python docfinqa_eval.py --evaluation-mode closed_book --limit 5

Test context window mode (full context in prompt):
    python docfinqa_eval.py --evaluation-mode context_window --limit 5

File search with query rewriting:
    python docfinqa_eval.py --evaluation-mode file_search --rewrite-queries --limit 5

File search with search results included:
    python docfinqa_eval.py --evaluation-mode file_search --include-search-results --limit 5

Compare all modes on same dataset:
    python docfinqa_eval.py --evaluation-mode file_search --limit 10 --output results_file_search.json
    python docfinqa_eval.py --evaluation-mode closed_book --limit 10 --output results_closed_book.json
    python docfinqa_eval.py --evaluation-mode context_window --limit 10 --output results_context_window.json

Use different model:
    python docfinqa_eval.py --model gpt-4o-mini --evaluation-mode closed_book --limit 10
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
    level=getattr(logging, os.getenv('OPENAI_LOG_LEVEL', 'WARNING')),  # Changed from INFO to WARNING
    format='%(asctime)s - %(levelname)s - %(message)s'              # Log format with timestamp
)
logger = logging.getLogger(__name__)  # Create logger for this script

# Suppress OpenAI and HTTP request logging unless specifically requested
if not os.getenv('OPENAI_LOG_LEVEL'):
    logging.getLogger('openai').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    # Also suppress any other noisy loggers
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)

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
    - Search results (if enabled)
    
    Example:
        If we asked "What was Q3 revenue?" and expected "100":
        - model_response might be "FINAL: 100.0"
        - numerical_accuracy would be 1.0 (perfect match)
        - extracted_number would be 100.0
        - success would be True
        - search_results might contain the document chunks that were found
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
    
    # === SEARCH INFORMATION ===
    # These fields contain information about the search process
    rewritten_query: Optional[str] = None       # Query after rewriting (if used)
    search_results: Optional[List[Dict]] = None # File search results (if enabled)
    
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
                 retry_delay: float = 1.0,
                 use_query_rewriting: bool = False,
                 include_search_results: bool = False,
                 evaluation_mode: str = "file_search"):
        """
        Initialize the evaluator - sets up connection to OpenAI and configuration
        
        Args:
            api_key: Your OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: Which AI model to use (gpt-4o, gpt-4o-mini, etc.)
            max_retries: How many times to retry if API calls fail
            retry_delay: How long to wait between retries (in seconds)
            use_query_rewriting: Whether to rewrite queries before file search (only for file_search mode)
            include_search_results: Whether to include file search results in response (only for file_search mode)
            evaluation_mode: Evaluation mode - "file_search", "closed_book", or "context_window"
            
        Example:
            evaluator = DocFinQAEvaluator(model="gpt-4o-mini", evaluation_mode="closed_book")
            evaluator = DocFinQAEvaluator(model="gpt-4o-mini", evaluation_mode="context_window")
            evaluator = DocFinQAEvaluator(model="gpt-4o-mini", evaluation_mode="file_search", use_query_rewriting=True)
        """
        # Validate evaluation mode
        valid_modes = ["file_search", "closed_book", "context_window"]
        if evaluation_mode not in valid_modes:
            raise ValueError(f"evaluation_mode must be one of {valid_modes}, got: {evaluation_mode}")
            
        # Disable OpenAI client logging if not explicitly requested
        if not os.getenv('OPENAI_LOG_LEVEL'):
            # Set environment variable to suppress OpenAI HTTP logging
            os.environ['OPENAI_LOG_LEVEL'] = 'error'
            
        # Create OpenAI client - this is our connection to OpenAI's servers
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
        # Store configuration settings
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')  # AI model to use
        self.max_retries = max_retries      # How many times to retry failed calls
        self.retry_delay = retry_delay      # Seconds to wait between retries
        self.evaluation_mode = evaluation_mode  # Which evaluation mode to use
        
        # File search specific settings (only used when evaluation_mode is "file_search")
        self.use_query_rewriting = use_query_rewriting if evaluation_mode == "file_search" else False
        self.include_search_results = include_search_results if evaluation_mode == "file_search" else False
        
        # Log warnings if file_search-specific options are used with other modes
        if evaluation_mode != "file_search":
            if use_query_rewriting:
                logger.warning("Query rewriting is only available in file_search mode, ignoring --rewrite-queries")
            if include_search_results:
                logger.warning("Search results inclusion is only available in file_search mode, ignoring --include-search-results")
        
        print(f"Initialized DocFinQA Evaluator with model: {self.model}, mode: {self.evaluation_mode}, query rewriting: {self.use_query_rewriting}, include search results: {self.include_search_results}")
        
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
        print(f"Loading test data from {file_path}")
        
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
                
            print(f"Loaded {len(test_cases)} test cases")
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
                    # Removed verbose logging - file processing completed quietly
                    break
                elif vector_store_files.data and vector_store_files.data[0].status == 'failed':
                    raise RuntimeError(f"Vector store file processing failed: {vector_store_files.data[0].last_error}")
                    
                time.sleep(check_interval)
                wait_time += check_interval
                
                # Log progress every 30 seconds
                if wait_time % 30 == 0:
                    print(f"Still processing vector store file... ({wait_time}s elapsed)")
                
            if wait_time >= max_wait:
                raise TimeoutError(f"Vector store file processing timed out after {max_wait} seconds")
                
            # Removed verbose logging - vector store created quietly
            return vector_store.id
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    def rewrite_query(self, original_query: str) -> str:
        """
        Rewrite the original query to be more effective for financial document search
        
        Args:
            original_query: The original question from the test case
            
        Returns:
            Rewritten query optimized for document search
            
        Example:
            original: "what is the net change in net revenue during 2015?"
            rewritten: "net revenue change 2015 increase decrease financial statement"
        """
        try:
            # Create a prompt to rewrite the query for better document search
            rewrite_prompt = f"""You are a financial document search expert. Rewrite the following question to create better search terms for finding relevant information in financial documents.

Original question: {original_query}

Guidelines for rewriting:
1. Extract key financial terms, metrics, and time periods
2. Add relevant synonyms that might appear in financial documents
3. Focus on searchable keywords rather than question structure
4. Include related financial concepts that might help find the answer

Rewritten search query:"""

            # Use the Responses API to rewrite the query
            response = self.client.responses.create(
                model=self.model,
                input=rewrite_prompt
            )
            
            # Extract the rewritten query
            rewritten_query = ""
            for output in response.output:
                if hasattr(output, 'content') and output.content:
                    for content in output.content:
                        if hasattr(content, 'text'):
                            rewritten_query += content.text + " "
            
            rewritten_query = rewritten_query.strip()
            
            if rewritten_query:
                # Query rewriting successful - no verbose logging needed
                return rewritten_query
            else:
                logger.warning("Query rewriting failed, using original query")
                return original_query
                
        except Exception as e:
            logger.error(f"Error in query rewriting: {e}")
            return original_query
            
    def extract_search_results(self, response) -> Optional[List[Dict]]:
        """
        Extract file search results from the OpenAI response
        
        Based on OpenAI Responses API documentation, when using include=["output[*].file_search_call.search_results"],
        the search results are accessible through the response output structure.
        
        Args:
            response: OpenAI response object
            
        Returns:
            List of search result dictionaries, or None if not available
        """
        try:
            search_results = []
            # Removed verbose debug logging - extracting quietly
            
            # According to the documentation, when include=["output[*].file_search_call.search_results"] is used,
            # the search results should be available in the response output
            if hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    # Look for file_search_call type in output
                    if hasattr(output_item, 'type') and output_item.type == 'file_search_call':
                        # Check if there are results in this output (note: it's 'results', not 'search_results')
                        if hasattr(output_item, 'results'):
                            for result in output_item.results:
                                result_dict = self._extract_single_result(result)
                                if result_dict:
                                    search_results.append(result_dict)
                    
                    # Also check if this output has annotations (alternative way search results might be exposed)
                    elif hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'annotations') and content_item.annotations:
                                for annotation in content_item.annotations:
                                    if hasattr(annotation, 'filename'):  # This indicates it's a file search result
                                        result_dict = self._extract_single_result(annotation)
                                        if result_dict:
                                            search_results.append(result_dict)
            
            # If no results found through output, try alternative approach
            if not search_results:
                # Sometimes the include data might be in a separate attribute
                for attr_name in ['include', 'included', 'search_results', 'tool_results']:
                    if hasattr(response, attr_name):
                        attr_value = getattr(response, attr_name)
                        if attr_value:
                            self._deep_inspect_response(attr_value, depth=0, max_depth=2)
            
            # Only log if search results extraction fails
            if not search_results and self.include_search_results:
                logger.warning("No search results found in response - extraction failed")
                
            return search_results if search_results else None
            
        except Exception as e:
            logger.error(f"Error extracting search results: {e}")
            return None
    
    def _extract_single_result(self, result) -> Optional[Dict]:
        """Extract a single search result into a dictionary"""
        try:
            result_dict = {}
            
            # Try different possible attribute names for file information
            for attr_name in ['file_id', 'id', 'document_id']:
                if hasattr(result, attr_name):
                    result_dict['file_id'] = getattr(result, attr_name)
                    break
                    
            for attr_name in ['file_name', 'filename', 'name', 'title']:
                if hasattr(result, attr_name):
                    result_dict['file_name'] = getattr(result, attr_name)
                    break
                    
            for attr_name in ['content', 'text', 'snippet', 'excerpt']:
                if hasattr(result, attr_name):
                    content_val = getattr(result, attr_name)
                    # Handle content that might be a list or direct text
                    if isinstance(content_val, list) and len(content_val) > 0:
                        result_dict['content'] = content_val[0].get('text', str(content_val[0])) if hasattr(content_val[0], 'get') else str(content_val[0])
                    else:
                        result_dict['content'] = str(content_val)
                    break
                    
            for attr_name in ['score', 'relevance', 'confidence']:
                if hasattr(result, attr_name):
                    result_dict['score'] = getattr(result, attr_name)
                    break
            
            # If we found any data, return it
            if result_dict:
                return result_dict
            else:
                return None
            
        except Exception as e:
            return None
    
    def _deep_inspect_response(self, obj, depth=0, max_depth=3):
        """Recursively inspect the response object to find search results"""
        if depth > max_depth:
            return
            
        # Simplified inspection - no verbose logging
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                if 'search' in attr_name.lower() or 'result' in attr_name.lower() or 'file' in attr_name.lower():
                    if depth < max_depth and attr_value is not None:
                        self._deep_inspect_response(attr_value, depth + 1, max_depth)
                
    def evaluate_single_case(self, test_case: TestCase, case_index: int) -> EvalResult:
        """
        Evaluate a single test case using the specified evaluation mode
        
        This method takes one financial question and document, then evaluates it using one of three modes:
        1. "file_search": Uploads document to vector store and uses file_search tool (original behavior)
        2. "closed_book": Asks the question without any context (tests base model knowledge)
        3. "context_window": Provides context directly in the prompt (tests reasoning with full context)
        
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
        
        # Initialize tracking variables
        vector_store_id = None  # Only used for file_search mode
        response_id = None      # The Responses API response ID
        
        try:
            print(f"Evaluating test case {case_index + 1}...")
            
            # Route to appropriate evaluation method based on mode
            if self.evaluation_mode == "file_search":
                return self._evaluate_file_search_mode(test_case, case_index, start_time)
            elif self.evaluation_mode == "closed_book":
                return self._evaluate_closed_book_mode(test_case, case_index, start_time)
            elif self.evaluation_mode == "context_window":
                return self._evaluate_context_window_mode(test_case, case_index, start_time)
            else:
                raise ValueError(f"Unknown evaluation mode: {self.evaluation_mode}")
                
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
            
    def _evaluate_file_search_mode(self, test_case: TestCase, case_index: int, start_time: float) -> EvalResult:
        """
        Evaluate using file_search mode (original behavior)
        
        1. Uploads the document to OpenAI as a searchable vector store
        2. Uses the Responses API with file_search tool to answer the question
        3. Scores the response for accuracy
        4. Cleans up all the temporary resources
        """
        vector_store_id = None
        response_id = None
        
        try:
            # === STEP 1: UPLOAD DOCUMENT ===
            # Create a vector store (searchable database) with the financial document
            vector_store_id = self.create_vector_store_with_context(
                test_case.context,                    # The financial document text
                f"docfinqa-case-{case_index}"        # Give it a unique name
            )
            
            # === STEP 2: OPTIONALLY REWRITE QUERY ===
            # If query rewriting is enabled, rewrite the question for better search
            search_question = test_case.question
            if self.use_query_rewriting:
                search_question = self.rewrite_query(test_case.question)
            
            # === STEP 3: USE RESPONSES API WITH FILE_SEARCH ===
            # Create a formatted prompt with the question and instructions
            if self.use_query_rewriting:
                prompt = f"""### Original Question
{test_case.question}

### Search Query Used
{search_question}

### Instructions
Use the file_search tool to find information from the uploaded financial document to answer the original question.
The search query above has been optimized for better document search.

Work step by step under the header REASONING: and explain your thinking.
Finish with a single line with only your numerical or logical answer under the header FINAL: <answer>
"""
            else:
                prompt = f"""### Question
{test_case.question}

### Instructions
Use the file_search tool to find information from the uploaded financial document to answer the question.

Work step by step under the header REASONING: and explain your thinking.
Finish with a single line with only your numerical or logical answer under the header FINAL: <answer>
"""
            
            # Prepare the API call parameters
            api_params = {
                "model": self.model,
                "input": prompt,
                "tools": [{
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id]
                }]
            }
            
            # Add include parameter if search results are requested
            # Based on the documentation, this should include search results in the response
            if self.include_search_results:
                api_params["include"] = ["output[*].file_search_call.search_results"]
            
            # Use the Responses API with file_search tool
            response = self.client.responses.create(**api_params)
            
            response_id = response.id
            
            # Extract the AI's response text
            model_response = ""
            for output in response.output:
                if hasattr(output, 'content') and output.content:
                    for content in output.content:
                        if hasattr(content, 'text'):
                            model_response += content.text + "\n"
            
            model_response = model_response.strip()
            
            # === STEP 4: EXTRACT SEARCH RESULTS (IF ENABLED) ===
            search_results = None
            rewritten_query = None
            
            if self.include_search_results:
                # Removed verbose debug logging
                search_results = self.extract_search_results(response)
                
            if self.use_query_rewriting:
                rewritten_query = search_question
            
            # === STEP 5: SCORE THE RESPONSE ===
            # Use our custom scoring system to evaluate the response
            scoring_result = score_response(model_response, test_case.expected_answer)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Removed verbose completion message
            
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
                found_final_tag=scoring_result.found_final_tag,
                # Search information (if enabled)
                rewritten_query=rewritten_query,
                search_results=search_results
            )
            
        finally:
            # === CLEANUP ===  
            # This runs whether we succeeded or failed
            # Always clean up OpenAI resources to avoid charges
            self.cleanup_resources(vector_store_id)
            
    def _evaluate_closed_book_mode(self, test_case: TestCase, case_index: int, start_time: float) -> EvalResult:
        """
        Evaluate using closed_book mode
        
        Asks the question without any context to test the model's base knowledge.
        This tests what the model knows about financial concepts without access to documents.
        """
        try:
            # Create prompt with just the question and instructions
            prompt = f"""### Question
{test_case.question}

### Instructions
Answer the question based on your knowledge. Do not use any external tools or search functions.

Work step by step under the header REASONING: and explain your thinking.
Finish with a single line with only your numerical or logical answer under the header FINAL: <answer>
"""
            
            # Use the Responses API without any tools
            response = self.client.responses.create(
                model=self.model,
                input=prompt
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
            
            # === SCORE THE RESPONSE ===
            # Use our custom scoring system to evaluate the response
            scoring_result = score_response(model_response, test_case.expected_answer)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Removed verbose completion message
            
            # Return successful result with scoring information
            return EvalResult(
                test_case=test_case,
                model_response=model_response,
                execution_time=execution_time,
                vector_store_id="",  # No vector store used
                response_id=response_id,
                success=True,
                # Scoring information from our custom scorer (ScoreResult object)
                numerical_accuracy=scoring_result.numerical_accuracy,
                extracted_number=scoring_result.extracted_number,
                expected_number=scoring_result.expected_number,
                found_final_tag=scoring_result.found_final_tag,
                # No search information in closed book mode
                rewritten_query=None,
                search_results=None
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            raise e  # Re-raise to be caught by main evaluate_single_case method
            
    def _evaluate_context_window_mode(self, test_case: TestCase, case_index: int, start_time: float) -> EvalResult:
        """
        Evaluate using context_window mode
        
        Provides the entire context directly in the prompt to test the model's reasoning
        ability when given full access to the financial document.
        """
        try:
            # Create prompt with context provided directly
            prompt = f"""### Context
<context>
{test_case.context}
</context>

### Question
{test_case.question}

### Instructions
Answer the question based on the context provided above. Do not use any external tools or search functions.

Work step by step under the header REASONING: and explain your thinking.
Finish with a single line with only your numerical or logical answer under the header FINAL: <answer>
"""
            
            # Use the Responses API without any tools
            response = self.client.responses.create(
                model=self.model,
                input=prompt
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
            
            # === SCORE THE RESPONSE ===
            # Use our custom scoring system to evaluate the response
            scoring_result = score_response(model_response, test_case.expected_answer)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Removed verbose completion message
            
            # Return successful result with scoring information
            return EvalResult(
                test_case=test_case,
                model_response=model_response,
                execution_time=execution_time,
                vector_store_id="",  # No vector store used
                response_id=response_id,
                success=True,
                # Scoring information from our custom scorer (ScoreResult object)
                numerical_accuracy=scoring_result.numerical_accuracy,
                extracted_number=scoring_result.extracted_number,
                expected_number=scoring_result.expected_number,
                found_final_tag=scoring_result.found_final_tag,
                # No search information in context window mode
                rewritten_query=None,
                search_results=None
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            raise e  # Re-raise to be caught by main evaluate_single_case method
            
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
                    # Removed verbose logging
                except AttributeError:
                    # Fall back to beta path for older versions
                    self.client.beta.vector_stores.delete(vector_store_id)
                    # Removed verbose logging
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
        
        print(f"Starting evaluation of {len(test_cases)} test cases")
        
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
            'evaluation_mode': self.evaluation_mode,
            'query_rewriting_enabled': self.use_query_rewriting,
            'search_results_included': self.include_search_results,
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
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only create directory if there's a path component
            os.makedirs(output_dir, exist_ok=True)
            
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
                'found_final_tag': result.found_final_tag,
                # Search information (if available)
                'rewritten_query': result.rewritten_query,
                'search_results': result.search_results
            }
            output_data['results'].append(result_data)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        print(f"Results saved to {output_file}")
        
    def save_incremental_results(self, results: List[EvalResult], output_file: str, total_expected: int = None):
        """
        Save results incrementally after each test case
        
        Args:
            results: List of evaluation results so far
            output_file: Path to output file
            total_expected: Total number of test cases expected (for progress tracking)
        """
        # Calculate partial metrics for current results
        partial_metrics = self.calculate_metrics(results)
        
        # Add progress information if total is known
        if total_expected:
            partial_metrics['progress'] = {
                'completed_cases': len(results),
                'total_cases': total_expected,
                'completion_percentage': (len(results) / total_expected) * 100,
                'is_complete': len(results) >= total_expected
            }
        
        # Save the results
        self.save_results(results, partial_metrics, output_file)
        
    def load_existing_results(self, output_file: str) -> List[EvalResult]:
        """
        Load existing results from a JSON file to enable resumption
        
        Args:
            output_file: Path to existing results file
            
        Returns:
            List of EvalResult objects from the existing file
        """
        if not os.path.exists(output_file):
            return []
            
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            results = []
            for result_data in data.get('results', []):
                # Reconstruct TestCase
                test_case = TestCase(
                    context="",  # We don't store context in results to save space
                    question=result_data['question'],
                    expected_answer=result_data['expected_answer'],
                    program=""
                )
                
                # Reconstruct EvalResult
                result = EvalResult(
                    test_case=test_case,
                    model_response=result_data['model_response'],
                    execution_time=result_data['execution_time'],
                    vector_store_id=result_data['vector_store_id'],
                    response_id=result_data['response_id'],
                    success=result_data['success'],
                    error_message=result_data['error_message'],
                    numerical_accuracy=result_data['numerical_accuracy'],
                    extracted_number=result_data['extracted_number'],
                    expected_number=result_data['expected_number'],
                    found_final_tag=result_data['found_final_tag'],
                    rewritten_query=result_data['rewritten_query'],
                    search_results=result_data['search_results']
                )
                results.append(result)
                
            print(f"Loaded {len(results)} existing results from {output_file}")
            return results
            
        except Exception as e:
            print(f"Warning: Could not load existing results from {output_file}: {e}")
            return []
            
    def find_completed_cases(self, existing_results: List[EvalResult], all_test_cases: List[TestCase]) -> set:
        """
        Find which test cases have already been completed
        
        Args:
            existing_results: Previously completed results
            all_test_cases: All test cases that need to be run
            
        Returns:
            Set of indices of completed test cases
        """
        completed_indices = set()
        
        # Create a map of questions to their indices in all_test_cases
        question_to_index = {case.question: i for i, case in enumerate(all_test_cases)}
        
        # Find which questions have been completed
        for result in existing_results:
            question = result.test_case.question
            if question in question_to_index:
                completed_indices.add(question_to_index[question])
                
        return completed_indices
        
    def evaluate_all_with_resume(self, test_cases: List[TestCase], output_file: str) -> List[EvalResult]:
        """
        Evaluate all test cases with automatic resume capability
        
        Args:
            test_cases: List of test cases to evaluate
            output_file: Path to output file (will be used for incremental saves)
            
        Returns:
            List of EvalResult objects
        """
        # Load any existing results
        existing_results = self.load_existing_results(output_file)
        completed_indices = self.find_completed_cases(existing_results, test_cases)
        
        # Start with existing results
        results = existing_results.copy()
        
        total_cases = len(test_cases)
        remaining_cases = total_cases - len(completed_indices)
        
        if completed_indices:
            print(f"Found {len(completed_indices)} already completed test cases")
            print(f"Resuming evaluation with {remaining_cases} remaining cases")
        else:
            print(f"Starting fresh evaluation of {total_cases} test cases")
        
        if remaining_cases == 0:
            print("All test cases already completed!")
            return results
            
        for i, test_case in enumerate(test_cases):
            # Skip if already completed
            if i in completed_indices:
                continue
                
            try:
                print(f"Evaluating test case {i + 1}/{total_cases} (remaining: {len(test_cases) - i - len([x for x in completed_indices if x <= i])})")
                
                result = self.evaluate_single_case(test_case, i)
                results.append(result)
                
                # Save incrementally after each test case
                try:
                    self.save_incremental_results(results, output_file, total_cases)
                except Exception as save_error:
                    print(f"Warning: Failed to save incremental results: {save_error}")
                
                # Brief pause between evaluations to avoid rate limits
                time.sleep(1)
                
            except KeyboardInterrupt:
                print(f"\n⚠️  Evaluation interrupted by user!")
                print(f"Progress saved to {output_file}")
                print(f"Completed {len(results)} out of {total_cases} test cases")
                print(f"To resume, run the same command again - it will continue from where it left off")
                break
                
            except Exception as e:
                logger.error(f"Fatal error on test case {i + 1}: {e}")
                # Save progress even after errors
                try:
                    self.save_incremental_results(results, output_file, total_cases)
                except Exception as save_error:
                    print(f"Warning: Failed to save incremental results after error: {save_error}")
                # Continue with next test case
                continue
                
        return results

def main():
    """Main entry point for the evaluation script"""
    # Create results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate default timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_output = os.path.join(results_dir, f'docfinqa_eval_{timestamp}.json')
    
    parser = argparse.ArgumentParser(description='DocFinQA Evaluation using OpenAI Responses API')
    parser.add_argument('--data', '-d', 
                       default='data/test-data-sample.json',
                       help='Path to test data JSON file')
    parser.add_argument('--output', '-o',
                       default=default_output,
                       help='Path to output results file')
    parser.add_argument('--limit', '-l',
                       type=int,
                       help='Limit number of test cases to process')
    parser.add_argument('--model', '-m',
                       default=None,
                       help='OpenAI model to use (overrides env var)')
    parser.add_argument('--rewrite-queries', 
                       action='store_true',
                       help='Enable query rewriting for better document search')
    parser.add_argument('--include-search-results', 
                       action='store_true',
                       help='Include file search results in the response and save them in the JSON output')
    parser.add_argument('--evaluation-mode', '-e',
                       default='file_search',
                       choices=['file_search', 'closed_book', 'context_window'],
                       help='Evaluation mode: "file_search" (uses file_search tool), "closed_book" (no context), or "context_window" (context in prompt)')
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.evaluation_mode != 'file_search':
        if args.rewrite_queries:
            print("Warning: --rewrite-queries only works with file_search mode")
        if args.include_search_results:
            print("Warning: --include-search-results only works with file_search mode")
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)
        
    try:
        # Initialize evaluator
        evaluator = DocFinQAEvaluator(
            model=args.model, 
            use_query_rewriting=args.rewrite_queries,
            include_search_results=args.include_search_results,
            evaluation_mode=args.evaluation_mode
        )
        
        # Load test data
        test_cases = evaluator.load_test_data(args.data, args.limit)
        
        if not test_cases:
            print("Error: No test cases loaded")
            sys.exit(1)
            
        # Run evaluation
        results = evaluator.evaluate_all_with_resume(test_cases, args.output)
        
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
        print(f"Error: Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 