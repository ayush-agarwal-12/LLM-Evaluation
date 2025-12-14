"""
Relevance and Completeness Evaluation Module
Evaluates how well AI responses address user queries
"""

import time
import json
import logging
from typing import Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class RelevanceEvaluator:
    """Evaluates response relevance and completeness"""
    
    def __init__(self, groq_client, model_name: str = "gpt-oss-120b"):
        """
        Initialize relevance evaluator
        
        Args:
            groq_client: Groq API client instance
            model_name: LLM model name for evaluation
        """
        self.client = groq_client
        self.model_name = model_name
        
        # Load embedding model for semantic similarity
        logger.info("Loading embedding model for relevance evaluation...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _get_default_relevance_result(self) -> Dict:
        """Return default result when LLM call fails"""
        return {
            "relevance_score": 5,
            "completeness_score": 5,
            "context_usage_score": 5,
            "reasoning": "Unable to evaluate - LLM response error",
            "tokens": {"input": 0, "output": 0, "total": 0}
        }
        
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using embeddings
        
        Args:
            text1: First text (typically user query)
            text2: Second text (typically AI response)
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def evaluate_with_llm(
        self,
        query: str,
        response: str,
        context: str
    ) -> Dict:
        """
        Use LLM to evaluate relevance and completeness
        
        Args:
            query: User's question/query
            response: AI's response to evaluate
            context: Retrieved context from vector DB
            
        Returns:
            Dictionary with LLM evaluation scores
        """
        # Truncate to avoid token limits
        context_preview = context[:1000] if len(context) > 1000 else context
        response_text = response[:800] if len(response) > 800 else response
        query_text = query[:300] if len(query) > 300 else query
        
        # Very explicit prompt - NO extra text
        prompt = f"""Evaluate this AI response. Return ONLY the JSON, no explanations.

Query: {query_text}
Response: {response_text}
Context: {context_preview}

Rate 0-10: Relevance, Completeness, Context usage.

Return ONLY this JSON (no other text):
{{"relevance_score": <0-10>, "completeness_score": <0-10>, "context_usage_score": <0-10>, "reasoning": "<text>"}}"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400
            )
            
            # Extract response and token usage
            llm_response = completion.choices[0].message.content
            usage = completion.usage
            
            logger.debug(f"Raw relevance response: {llm_response[:200]}...")
            
            # Check for empty response
            if not llm_response or llm_response.strip() == "":
                logger.warning("Empty response from relevance check, using fallback")
                return self._get_default_relevance_result()
            
            # Aggressive JSON extraction
            llm_response = llm_response.strip()
            
            # Remove markdown code blocks
            llm_response = llm_response.replace("```json", "").replace("```", "")
            
            # Remove common prefixes
            prefixes_to_remove = [
                "Here's the evaluation:",
                "Here's the JSON response:",
                "Here is the response:",
                "Evaluation:",
                "Response:",
                "Here's",
                "Here is"
            ]
            for prefix in prefixes_to_remove:
                if llm_response.strip().startswith(prefix):
                    llm_response = llm_response.strip()[len(prefix):].strip()
            
            # Find the JSON object
            if "{" in llm_response and "}" in llm_response:
                start = llm_response.find("{")
                end = llm_response.rfind("}") + 1
                json_str = llm_response[start:end].strip()
                
                # Parse JSON
                eval_result = json.loads(json_str)
                
                return {
                    "relevance_score": eval_result.get("relevance_score", 5),
                    "completeness_score": eval_result.get("completeness_score", 5),
                    "context_usage_score": eval_result.get("context_usage_score", 5),
                    "reasoning": eval_result.get("reasoning", "Evaluation completed"),
                    "tokens": {
                        "input": usage.prompt_tokens,
                        "output": usage.completion_tokens,
                        "total": usage.total_tokens
                    }
                }
            else:
                logger.error("No JSON braces found in response")
                return self._get_default_relevance_result()
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse relevance response: {e}")
            if 'llm_response' in locals():
                logger.error(f"Raw response was: '{llm_response[:300]}'")
            else:
                logger.error("No response received from API")
            
            # Try regex fallback
            try:
                if 'llm_response' in locals() and llm_response:
                    import re
                    rel_match = re.search(r'"?relevance_score"?\s*:\s*(\d+)', llm_response)
                    comp_match = re.search(r'"?completeness_score"?\s*:\s*(\d+)', llm_response)
                    ctx_match = re.search(r'"?context_usage_score"?\s*:\s*(\d+)', llm_response)
                    reason_match = re.search(r'"?reasoning"?\s*:\s*"([^"]+)"', llm_response)
                    
                    if rel_match and comp_match:
                        logger.info("Extracted scores via regex fallback")
                        return {
                            "relevance_score": int(rel_match.group(1)),
                            "completeness_score": int(comp_match.group(1)),
                            "context_usage_score": int(ctx_match.group(1)) if ctx_match else 5,
                            "reasoning": reason_match.group(1) if reason_match else "Partial evaluation (regex fallback)",
                            "tokens": {"input": 0, "output": 0, "total": 0}
                        }
            except Exception as regex_error:
                logger.error(f"Regex fallback also failed: {regex_error}")
            
            return self._get_default_relevance_result()
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return self._get_default_relevance_result()
    
    def evaluate(
        self,
        query: str,
        response: str,
        context: str
    ) -> Dict:
        """
        Perform complete relevance evaluation
        
        Args:
            query: User query
            response: AI response
            context: Retrieved context
            
        Returns:
            Complete relevance evaluation results
        """
        start_time = time.time()
        
        # Calculate semantic similarity (fast, local)
        semantic_score = self.calculate_semantic_similarity(query, response)
        
        # Get LLM-based evaluation (more comprehensive)
        llm_eval = self.evaluate_with_llm(query, response, context)
        
        latency = time.time() - start_time
        
        # Calculate composite score (weighted average)
        composite_score = (
            semantic_score * 10 * 0.2 +  # Semantic similarity (20% weight)
            llm_eval['relevance_score'] * 0.4 +  # Relevance (40% weight)
            llm_eval['completeness_score'] * 0.4  # Completeness (40% weight)
        )
        
        return {
            "semantic_similarity": round(semantic_score, 3),
            "llm_relevance_score": llm_eval['relevance_score'],
            "llm_completeness_score": llm_eval['completeness_score'],
            "context_usage_score": llm_eval['context_usage_score'],
            "composite_score": round(composite_score, 2),
            "reasoning": llm_eval['reasoning'],
            "tokens_used": llm_eval['tokens'],
            "latency_seconds": round(latency, 3)
        }