"""
Performance Metrics Module
Tracks latency and cost metrics for evaluations
"""

import time
import logging
from typing import Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Track token usage across evaluations"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, input_tokens: int, output_tokens: int):
        """Add tokens to the counter"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += (input_tokens + output_tokens)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class CostCalculator:
    """Calculate costs based on token usage"""
    
    # Groq pricing (per 1M tokens)
    input_price_per_1m: float = 0.59
    output_price_per_1m: float = 0.79
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for given token usage
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * self.input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_1m
        return input_cost + output_cost
    
    def calculate_from_usage(self, usage: TokenUsage) -> float:
        """Calculate cost from TokenUsage object"""
        return self.calculate_cost(usage.input_tokens, usage.output_tokens)


class PerformanceTracker:
    """Track performance metrics for evaluations"""
    
    def __init__(self):
        """Initialize performance tracker"""
        self.token_usage = TokenUsage()
        self.cost_calculator = CostCalculator()
        
        self.evaluation_times: List[float] = []
        self.relevance_times: List[float] = []
        self.hallucination_times: List[float] = []
        
        self.start_time = None
        self.end_time = None
    
    def start_tracking(self):
        """Start tracking overall time"""
        self.start_time = time.time()
    
    def stop_tracking(self):
        """Stop tracking overall time"""
        self.end_time = time.time()
    
    def add_tokens(self, tokens_dict: Dict):
        """
        Add token usage from evaluation
        
        Args:
            tokens_dict: Dictionary with 'input' and 'output' keys
        """
        if tokens_dict and isinstance(tokens_dict, dict):
            input_tokens = tokens_dict.get('input', 0)
            output_tokens = tokens_dict.get('output', 0)
            self.token_usage.add(input_tokens, output_tokens)
    
    def add_evaluation_time(self, latency: float):
        """Add time for a complete evaluation"""
        self.evaluation_times.append(latency)
    
    def add_relevance_time(self, latency: float):
        """Add time for relevance evaluation"""
        self.relevance_times.append(latency)
    
    def add_hallucination_time(self, latency: float):
        """Add time for hallucination detection"""
        self.hallucination_times.append(latency)
    
    def get_total_cost(self) -> float:
        """Get total cost of all evaluations"""
        return self.cost_calculator.calculate_from_usage(self.token_usage)
    
    def get_average_latency(self) -> float:
        """Get average evaluation latency"""
        if not self.evaluation_times:
            return 0.0
        return sum(self.evaluation_times) / len(self.evaluation_times)
    
    def get_total_time(self) -> float:
        """Get total elapsed time"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def get_metrics_summary(self) -> Dict:
        """
        Get comprehensive performance metrics summary
        
        Returns:
            Dictionary with all performance metrics
        """
        total_evaluations = len(self.evaluation_times)
        total_cost = self.get_total_cost()
        avg_latency = self.get_average_latency()
        total_time = self.get_total_time()
        
        # Calculate throughput (evaluations per second)
        throughput = 0.0
        if total_time > 0:
            throughput = total_evaluations / total_time
        
        return {
            "total_evaluations": total_evaluations,
            "total_time_seconds": round(total_time, 3),
            "average_latency_seconds": round(avg_latency, 3),
            "throughput_per_second": round(throughput, 3),
            "token_usage": self.token_usage.to_dict(),
            "total_cost_usd": round(total_cost, 6),
            "cost_per_evaluation": round(total_cost / total_evaluations, 6) if total_evaluations > 0 else 0,
            "breakdown": {
                "relevance_avg_seconds": round(sum(self.relevance_times) / len(self.relevance_times), 3) if self.relevance_times else 0,
                "hallucination_avg_seconds": round(sum(self.hallucination_times) / len(self.hallucination_times), 3) if self.hallucination_times else 0
            }
        }
    
    def log_summary(self):
        """Log performance summary"""
        summary = self.get_metrics_summary()
        logger.info("=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Evaluations: {summary['total_evaluations']}")
        logger.info(f"Total Time: {summary['total_time_seconds']}s")
        logger.info(f"Average Latency: {summary['average_latency_seconds']}s")
        logger.info(f"Throughput: {summary['throughput_per_second']} eval/s")
        logger.info(f"Total Tokens: {summary['token_usage']['total_tokens']}")
        logger.info(f"Total Cost: ${summary['total_cost_usd']}")
        logger.info(f"Cost per Eval: ${summary['cost_per_evaluation']}")
        logger.info("=" * 60)


class LatencyMonitor:
    """Context manager for monitoring latency of operations"""
    
    def __init__(self, operation_name: str = "operation"):
        """
        Initialize latency monitor
        
        Args:
            operation_name: Name of operation being monitored
        """
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.latency = None
    
    def __enter__(self):
        """Start timing"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and calculate latency"""
        self.end_time = time.time()
        self.latency = self.end_time - self.start_time
        logger.debug(f"{self.operation_name} took {self.latency:.3f}s")
    
    def get_latency(self) -> float:
        """Get measured latency"""
        return self.latency if self.latency else 0.0


def estimate_scale_metrics(
    avg_latency: float,
    cost_per_eval: float,
    daily_conversations: int,
    sampling_rate: float = 0.1
) -> Dict:
    """
    Estimate metrics at scale
    
    Args:
        avg_latency: Average latency per evaluation (seconds)
        cost_per_eval: Cost per evaluation (USD)
        daily_conversations: Number of daily conversations
        sampling_rate: Percentage of conversations to evaluate (0-1)
        
    Returns:
        Dictionary with scale estimates
    """
    daily_evaluations = int(daily_conversations * sampling_rate)
    
    # Calculate required capacity
    seconds_per_day = 86400
    evaluations_per_instance = seconds_per_day / avg_latency
    required_instances = daily_evaluations / evaluations_per_instance
    
    # Calculate costs
    daily_cost = daily_evaluations * cost_per_eval
    monthly_cost = daily_cost * 30
    
    return {
        "daily_conversations": daily_conversations,
        "sampling_rate": sampling_rate,
        "daily_evaluations": daily_evaluations,
        "evaluations_per_instance_per_day": int(evaluations_per_instance),
        "required_instances": round(required_instances, 2),
        "recommended_instances": int(required_instances * 1.5),  # 50% buffer
        "costs": {
            "per_evaluation": round(cost_per_eval, 6),
            "daily": round(daily_cost, 2),
            "monthly": round(monthly_cost, 2),
            "yearly": round(daily_cost * 365, 2)
        },
        "performance": {
            "avg_latency_seconds": round(avg_latency, 3),
            "max_throughput_per_instance": round(1 / avg_latency, 3)
        }
    }