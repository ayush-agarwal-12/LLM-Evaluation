"""
Helper Utilities
Common functions for data processing and formatting
"""

import json
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def save_json_file(data: Dict, file_path: str, indent: int = 2):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        file_path: Output file path
        indent: JSON indentation level
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Successfully saved results to {file_path}")
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {e}")
        raise


def extract_messages_from_conversation(conversation: Dict) -> List[Dict]:
    """
    Extract messages from conversation JSON
    
    Args:
        conversation: Conversation dictionary
        
    Returns:
        List of message dictionaries
    """
    # Handle different conversation structures
    if 'messages' in conversation:
        messages = conversation['messages']
    elif 'conversation_turns' in conversation:
        messages = conversation['conversation_turns']
    elif isinstance(conversation, list):
        messages = conversation
    else:
        messages = []
    
    if not messages:
        logger.warning("No messages found in conversation")
    else:
        logger.info(f"Extracted {len(messages)} messages from conversation")
    
    return messages


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_timestamp(timestamp: Any = None) -> str:
    """
    Format timestamp to readable string
    
    Args:
        timestamp: Timestamp (string or datetime object)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if isinstance(timestamp, str):
        return timestamp
    
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    return str(timestamp)


def calculate_statistics(scores: List[float]) -> Dict:
    """
    Calculate statistical metrics for a list of scores
    
    Args:
        scores: List of numerical scores
        
    Returns:
        Dictionary with statistics
    """
    if not scores:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std_dev": 0.0,
            "count": 0
        }
    
    scores_sorted = sorted(scores)
    n = len(scores)
    mean_val = sum(scores) / n
    
    # Calculate median
    if n % 2 == 0:
        median_val = (scores_sorted[n//2 - 1] + scores_sorted[n//2]) / 2
    else:
        median_val = scores_sorted[n//2]
    
    # Calculate standard deviation
    variance = sum((x - mean_val) ** 2 for x in scores) / n
    std_dev = variance ** 0.5
    
    return {
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "min": round(min(scores), 2),
        "max": round(max(scores), 2),
        "std_dev": round(std_dev, 2),
        "count": n
    }


def format_evaluation_report(
    results: List[Dict],
    summary: Dict,
    performance: Dict
) -> str:
    """
    Format evaluation results into readable text report
    
    Args:
        results: Evaluation results
        summary: Summary statistics
        performance: Performance metrics
        
    Returns:
        Formatted report string
    """
    report_lines = [
        "=" * 80,
        "LLM EVALUATION REPORT",
        "=" * 80,
        "",
        f"Generated: {format_timestamp()}",
        f"Total Responses Evaluated: {len(results)}",
        "",
        "QUALITY METRICS:",
        "-" * 80,
        f"Average Relevance Score: {summary['relevance_scores']['mean']}/10",
        f"Average Completeness Score: {summary['completeness_scores']['mean']}/10",
        f"Average Faithfulness Score: {summary['hallucination_scores']['mean']}/10",
        f"Overall Quality: {summary['overall_quality']['score']}/10 ({summary['overall_quality']['rating']})",
        "",
        "PERFORMANCE METRICS:",
        "-" * 80,
        f"Total Time: {performance['total_time_seconds']}s",
        f"Average Latency: {performance['average_latency_seconds']}s",
        f"Total Tokens: {performance['token_usage']['total_tokens']}",
        f"Total Cost: ${performance['total_cost_usd']}",
        f"Cost per Evaluation: ${performance['cost_per_evaluation']}",
        "",
        "=" * 80
    ]
    
    return "\n".join(report_lines)


def filter_responses_by_criteria(
    results: List[Dict],
    min_relevance: float = 0.0,
    min_faithfulness: float = 0.0
) -> List[Dict]:
    """
    Filter evaluation results by quality criteria
    
    Args:
        results: List of evaluation results
        min_relevance: Minimum relevance score threshold
        min_faithfulness: Minimum faithfulness score threshold
        
    Returns:
        Filtered list of results
    """
    filtered = []
    
    for result in results:
        rel_score = result.get('relevance_evaluation', {}).get('llm_relevance_score', 0)
        faith_score = result.get('faithfulness_evaluation', {}).get('faithfulness_score', 0)
        
        if rel_score >= min_relevance and faith_score >= min_faithfulness:
            filtered.append(result)
    
    logger.info(f"Filtered {len(results)} results to {len(filtered)} based on criteria")
    return filtered