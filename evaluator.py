"""
LLM Response Evaluation Pipeline - Main Script
Now handles broken/malformed JSON files
"""

import os
import sys
import logging
from typing import Dict, Optional, Tuple
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

# Import custom modules
from metrics.relevance import RelevanceEvaluator
from metrics.hallucination import HallucinationDetector
from metrics.performance import PerformanceTracker
from utils.dirty_loader import RobustDataLoader
from utils.helpers import (
    save_json_file,
    extract_messages_from_conversation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evaluation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def extract_latest_turn(conversation_data: Dict) -> Optional[Tuple[Dict, Dict]]:
    """
    Extract the LAST AI response and its immediately preceding user query.
    
    Args:
        conversation_data: Conversation dictionary
        
    Returns:
        Tuple of (user_message, ai_message) or None if not found
    """
    messages = extract_messages_from_conversation(conversation_data)
    
    if not messages:
        logger.warning("No messages found in conversation")
        return None
    
    # Iterate BACKWARDS to find the last AI response
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        sender = msg.get('sender', msg.get('role', msg.get('sender_id', ''))).lower()
        
        # Check if this is an AI message
        if 'ai' in sender or 'chatbot' in sender or sender == '1':
            logger.info(f"Found last AI response at index {i}")
            
            # Find the immediately preceding user message
            for j in range(i - 1, -1, -1):
                prev_msg = messages[j]
                prev_sender = prev_msg.get('sender', prev_msg.get('role', prev_msg.get('sender_id', ''))).lower()
                
                if 'user' in prev_sender or prev_sender == '0' or (prev_sender != '1' and 'ai' not in prev_sender and 'chatbot' not in prev_sender):
                    logger.info(f"Found corresponding user query at index {j}")
                    return (prev_msg, msg)
            
            logger.warning(f"Found AI response at index {i} but no preceding user message")
            return None
    
    logger.warning("No AI responses found in conversation")
    return None


class LLMEvaluationPipeline:
    """Main pipeline orchestrator for LLM response evaluation (ROBUST VERSION)"""
    
    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize the evaluation pipeline
        
        Args:
            groq_api_key: Groq API key
            model_name: LLM model to use for evaluations
        """
        logger.info("Initializing LLM Evaluation Pipeline (ROBUST MODE)...")
        
        # Initialize Groq client
        self.client = Groq(api_key=groq_api_key)
        self.model_name = model_name
        
        # Initialize robust data loader
        self.data_loader = RobustDataLoader()
        
        # Initialize evaluation modules
        logger.info("Loading evaluation modules...")
        self.relevance_evaluator = RelevanceEvaluator(self.client, model_name)
        self.hallucination_detector = HallucinationDetector(self.client, model_name)
        self.performance_tracker = PerformanceTracker()
        
        logger.info("Pipeline initialized successfully")
    
    def evaluate_single_response(
        self,
        query: str,
        response: str,
        context: str,
        message_id: str
    ) -> Dict:
        """
        Evaluate a single AI response
        
        Args:
            query: User query
            response: AI response
            context: Retrieved context
            message_id: Message identifier
            
        Returns:
            Complete evaluation results for the response
        """
        logger.info(f"Evaluating message: {message_id}")
        
        # Evaluate relevance and completeness
        relevance_result = self.relevance_evaluator.evaluate(
            query=query,
            response=response,
            context=context
        )
        
        # Track tokens and time
        self.performance_tracker.add_tokens(relevance_result['tokens_used'])
        self.performance_tracker.add_relevance_time(relevance_result['latency_seconds'])
        
        # Evaluate faithfulness (formerly hallucination)
        faithfulness_result = self.hallucination_detector.evaluate(
            response=response,
            context=context,
            check_contradictions=True
        )
        
        # Track tokens and time
        self.performance_tracker.add_tokens(faithfulness_result['tokens_used']['total'])
        self.performance_tracker.add_hallucination_time(faithfulness_result['latency_seconds'])
        
        # Calculate total evaluation time
        total_latency = (
            relevance_result['latency_seconds'] +
            faithfulness_result['latency_seconds']
        )
        self.performance_tracker.add_evaluation_time(total_latency)
        
        # Compile results with renamed faithfulness metric
        evaluation_result = {
            "message_id": message_id,
            "query": query,
            "response": response,
            "relevance_evaluation": relevance_result,
            "faithfulness_evaluation": {
                "faithfulness_score": faithfulness_result['hallucination_score'],
                "risk_level": faithfulness_result['risk_level'],
                "total_claims": faithfulness_result['total_claims'],
                "supported_claims": faithfulness_result['supported_claims'],
                "unsupported_claims": faithfulness_result['unsupported_claims'],
                "confidence": faithfulness_result['confidence'],
                "analysis": faithfulness_result['analysis'],
                "latency_seconds": faithfulness_result['latency_seconds'],
                "tokens_used": faithfulness_result['tokens_used']
            },
            "total_evaluation_time_seconds": round(total_latency, 3)
        }
        
        return evaluation_result
    
    def evaluate_latest_turn(
        self,
        conversation_path: str,
        context_path: str,
        output_path: str = "evaluation_results.json"
    ) -> Dict:
        """
        Evaluate ONLY the latest user query + AI response pair (ROBUST VERSION)
        
        Args:
            conversation_path: Path to conversation JSON file
            context_path: Path to context JSON file
            output_path: Path to save results
            
        Returns:
            Complete evaluation report
        """
        logger.info("=" * 80)
        logger.info("Starting LLM Evaluation Pipeline (ROBUST MODE - LATEST TURN ONLY)")
        logger.info("=" * 80)
        
        # Start performance tracking
        self.performance_tracker.start_tracking()
        
        try:
            # Load input data using ROBUST loader
            logger.info("\n" + "=" * 80)
            logger.info("LOADING DATA (ROBUST MODE)")
            logger.info("=" * 80)
            
            # Load conversation (tries JSON first, regex fallback)
            conversation = self.data_loader.load_conversation(conversation_path)
            
            # Load context (always uses dirty extraction - no JSON parsing)
            context_text = self.data_loader.load_context(context_path)
            
            # Validate loaded data
            is_valid, validation_msg = self.data_loader.validate_loaded_data(
                conversation, context_text
            )
            
            if not is_valid:
                raise ValueError(f"Invalid input data: {validation_msg}")
            
            logger.info(" Input data loaded and validated successfully")
            
            # Extract ONLY the latest turn
            logger.info("\n" + "=" * 80)
            logger.info("EXTRACTING LATEST TURN (User Query + AI Response)")
            logger.info("=" * 80)
            
            turn_pair = extract_latest_turn(conversation)
            
            if not turn_pair:
                logger.error("No valid AI response found in conversation")
                return {
                    "error": "No AI responses found in conversation",
                    "evaluation_result": None,
                    "summary": {}
                }
            
            user_msg, ai_msg = turn_pair
            
            # Extract text content
            query = user_msg.get('message', '')
            response = ai_msg.get('message', '')
            message_id = str(ai_msg.get('id') or ai_msg.get('message_id') or ai_msg.get('turn') or 'latest')
            
            # VISIBILITY: Print the identified turn
            print("\n" + "=" * 80)
            print("IDENTIFIED LATEST TURN FOR EVALUATION")
            print("=" * 80)
            print(f"\n USER QUERY (ID: {user_msg.get('id', 'N/A')}):")
            print("-" * 80)
            print(query[:500] + "..." if len(query) > 500 else query)
            print(f"\n AI RESPONSE (ID: {message_id}):")
            print("-" * 80)
            print(response[:500] + "..." if len(response) > 500 else response)
            print("\n" + "=" * 80 + "\n")
            
            if not query or not response:
                logger.error("Empty query or response found")
                return {
                    "error": "Empty query or response in latest turn",
                    "evaluation_result": None,
                    "summary": {}
                }
            
            # Context is already loaded as plain text from DirtyLoader
            logger.info(f"Context available: {len(context_text)} characters")
            
            # Evaluate ONCE - the latest turn only
            logger.info("\n" + "=" * 80)
            logger.info("RUNNING EVALUATION ON LATEST TURN")
            logger.info("=" * 80)
            
            result = self.evaluate_single_response(
                query=query,
                response=response,
                context=context_text,
                message_id=message_id
            )
            
            # Stop performance tracking
            self.performance_tracker.stop_tracking()
            
            # Log results
            logger.info("\n" + "=" * 80)
            logger.info("EVALUATION RESULTS")
            logger.info("=" * 80)
            logger.info(f"  Relevance Score: {result['relevance_evaluation']['llm_relevance_score']}/10")
            logger.info(f"  Completeness Score: {result['relevance_evaluation']['llm_completeness_score']}/10")
            logger.info(f"  Faithfulness Score: {result['faithfulness_evaluation']['faithfulness_score']}/10")
            logger.info(f"  Risk Level: {result['faithfulness_evaluation']['risk_level']}")
            logger.info("=" * 80)
            
            # Generate summary
            performance_metrics = self.performance_tracker.get_metrics_summary()
            
            # Combine results
            final_report = {
                "metadata": {
                    "conversation_file": conversation_path,
                    "context_file": context_path,
                    "model_used": self.model_name,
                    "evaluation_mode": "latest_turn_only",
                    "loading_mode": "robust_dirty_extraction",
                    "message_id": message_id
                },
                "evaluation_result": result,
                "summary": {
                    "relevance_score": result['relevance_evaluation']['llm_relevance_score'],
                    "completeness_score": result['relevance_evaluation']['llm_completeness_score'],
                    "faithfulness_score": result['faithfulness_evaluation']['faithfulness_score'],
                    "semantic_similarity": result['relevance_evaluation']['semantic_similarity'],
                    "risk_level": result['faithfulness_evaluation']['risk_level']
                },
                "performance": performance_metrics
            }
            
            # Save results
            logger.info(f"\nSaving results to {output_path}...")
            save_json_file(final_report, output_path)
            
            # Log performance summary
            self.performance_tracker.log_summary()
            
            # Print formatted report
            print("\n" + "=" * 80)
            print(" FINAL EVALUATION REPORT")
            print("=" * 80)
            print(f"Relevance Score:     {result['relevance_evaluation']['llm_relevance_score']}/10")
            print(f"Completeness Score:  {result['relevance_evaluation']['llm_completeness_score']}/10")
            print(f"Faithfulness Score:  {result['faithfulness_evaluation']['faithfulness_score']}/10")
            print(f"Semantic Similarity: {result['relevance_evaluation']['semantic_similarity']}")
            print(f"Risk Level:          {result['faithfulness_evaluation']['risk_level']}")
            print(f"\nTotal Tokens:        {performance_metrics['token_usage']['total_tokens']}")
            print(f"Total Cost:          ${performance_metrics['total_cost_usd']}")
            print(f"Evaluation Time:     {result['total_evaluation_time_seconds']}s")
            print("=" * 80)
            
            logger.info("\n Evaluation complete!")
            
            return final_report
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            raise
        finally:
            self.performance_tracker.stop_tracking()


def main():
    """Main execution function"""
    
    # Get API key from environment
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        print("\n Error: Please set GROQ_API_KEY environment variable")
        print("\nOn Windows (CMD):")
        print("  set GROQ_API_KEY=your_api_key_here")
        print("\nOn Windows (PowerShell):")
        print("  $env:GROQ_API_KEY='your_api_key_here'")
        print("\nOn macOS/Linux:")
        print("  export GROQ_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Configuration
    conversation_file = "conversation.json"
    context_file = "context.json"
    output_file = "evaluation_results.json"
    
    # Check if input files exist
    if not os.path.exists(conversation_file):
        logger.error(f"Conversation file not found: {conversation_file}")
        print(f"\n Error: {conversation_file} not found")
        print("Please ensure the conversation JSON file is in the current directory")
        sys.exit(1)
    
    if not os.path.exists(context_file):
        logger.error(f"Context file not found: {context_file}")
        print(f"\n Error: {context_file} not found")
        print("Please ensure the context JSON file is in the current directory")
        sys.exit(1)
    
    try:
        # Initialize pipeline with ROBUST mode
        print("\n Initializing ROBUST Evaluation Pipeline...")
        print("Can handle broken/malformed JSON")
        print("Uses regex fallback extraction")
        print("No JSON parsing for large context files\n")
        
        pipeline = LLMEvaluationPipeline(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile"
        )
        
        # Run evaluation on LATEST TURN ONLY
        results = pipeline.evaluate_latest_turn(
            conversation_path=conversation_file,
            context_path=context_file,
            output_path=output_file
        )
        
        print(f"\n Results saved to: {output_file}")
        print(f" Logs saved to: evaluation.log")
        
    except KeyboardInterrupt:
        logger.info("\n\n  Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\n Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()