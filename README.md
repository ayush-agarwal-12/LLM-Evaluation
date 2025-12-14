# LLM Response Evaluation Pipeline

Automated evaluation system for RAG chatbot responses using LLM-as-a-Judge methodology.

---

## Quick Start

### Prerequisites
- Python 3.8+
- Groq API key ([Get one here](https://console.groq.com/keys))

### Installation

```bash
pip install -r requirements.txt

# Set API key
export GROQ_API_KEY='your_api_key_here'  # Linux/Mac
set GROQ_API_KEY=your_api_key_here       # Windows CMD
```

### Run Evaluation

```bash
python evaluator.py
```

**Output:**
- Console: Evaluation scores
- `evaluation_results.json`: Full results
- `evaluation.log`: Detailed logs

---

## File Structure
```
├── evaluator.py              # Main orchestrator
├── conversation.json         # Chat history
├── context.json              # Retrieved context chunks
├── requirements.txt          # Project dependencies
├── metrics/
│   ├── relevance.py          # Relevance & completeness
│   ├── hallucination.py      # Faithfulness detection
│   └── performance.py        # Cost tracking
└── utils/
    ├── helpers.py            # Data processing
    └── dirty_loader.py       # Robust JSON handler
```

---

## Data Robustness

### Philosophy: Valid JSON First, Tolerance Second

**Best Practice:** Always use properly formatted, valid JSON files. The pipeline is optimized for clean data and will perform best with well-structured inputs.

**However**, real-world production systems sometimes produce imperfect data. The system can tolerate minor issues:

**What Can Be Auto-Fixed:**
- Trailing commas: `{"key": "value",}` → `{"key": "value"}`
- Extra whitespace and newlines
- Inconsistent key names (`messages` vs `conversation_turns`)
- Large files (100MB+) via regex extraction

**What Cannot Be Fixed:**
- Severely corrupted JSON (multiple syntax errors)
- Missing critical data (no messages, no context)
- Empty or incomplete responses

### Manual Intervention Required

**Important:** During testing, the assessment JSON file contained extensive formatting errors that exceeded the auto-repair capabilities. Manual cleanup was required before evaluation could proceed.

**Recommendation:** If your JSON files have complex structural issues:
1. Validate with `python -m json.tool your_file.json`
2. Use a JSON formatter/linter before pipeline ingestion
3. Fix structural issues at the source (data collection pipeline)

### 3-Tier Fallback System

```python
# Tier 1: Normal JSON parsing (fastest, preferred)
try:
    data = json.load(file)
except JSONDecodeError:
    # Tier 2: Auto-repair common issues
    fixed = clean_json(data)
    data = json.load(fixed)
except:
    # Tier 3: Regex extraction (last resort for context files only)
    context = extract_text_fields(raw_file)
```

**Key Innovation:** For context files specifically, text can be extracted without full JSON parsing when necessary.

```python
# Traditional (brittle):
context = json.load(f)['data']['vector_data'][0]['text']  # Fails on malformed JSON

# Fallback (robust):
pattern = r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"'
context = "\n\n".join(re.findall(pattern, raw_file))  # Works as last resort
```

**Tolerance Policy:**
- Minor JSON errors → Auto-fixed
- Major JSON corruption → Manual cleanup required
- Empty messages → Fail with clear error
- No AI responses → Skip evaluation

---

## Architecture

### Evaluation Pipeline

```
Input Files → Robust Loading → Turn Extraction → Metric Evaluation → Results
                (DirtyLoader)   (Latest turn)    (3 metrics)        (JSON output)
```

### Core Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| RobustDataLoader | Handle malformed JSON | Regex extraction + auto-repair |
| RelevanceEvaluator | Query-response alignment | Embeddings + LLM scoring |
| HallucinationDetector | Verify factual grounding | LLM claim verification |
| PerformanceTracker | Monitor costs | Token counting |

### Evaluation Metrics

1. **Relevance Score (0-10)**: Does the response address the query?
2. **Completeness Score (0-10)**: Are all aspects answered?
3. **Faithfulness Score (0-10)**: Is the response grounded in context?
4. **Semantic Similarity (0-1)**: Embedding-based alignment

---

## Key Design Decisions

### Why Evaluate Only Latest Turn?

**Reason:** Each RAG retrieval is turn-specific. Evaluating historical messages ("Hello", "Thanks") against the final context would produce false hallucinations.

**Technical Rationale:**
- `context.json` corresponds to the final retrieval step
- Historical turns have no relevant context
- Avoids wasting API calls on meaningless evaluations

### Why Custom JSON Loader?

**Problem:** Standard `json.load()` fails on production data with trailing commas, huge files, or inconsistent keys.

**Solution:** DirtyLoader uses graduated fallbacks. For large context files, regex extraction is 100x faster than JSON parsing.

### Why Groq API?

- 10x cheaper than GPT-4
- 5-10x faster inference
- Strong JSON output formatting

### Why Hybrid Scoring?

- **Embeddings**: Fast, cheap semantic similarity
- **LLM**: Understands context and nuance
- **Combined**: Balances speed and accuracy

---

## Performance Metrics

### Current Performance (Single Evaluation)
- Latency: ~2.6 seconds
- Tokens: ~2,500 per evaluation
- Cost: ~$0.00174 per evaluation

### Scale: 1M Daily Conversations

| Sampling Rate | Daily Cost | Monthly Cost | Instances Needed |
|---------------|------------|--------------|------------------|
| 100% | $1,740 | $52,200 | 12 |
| 10% | $174 | $5,220 | 2 |
| 1% | $17.40 | $522 | 1 |

### Latency Breakdown

| Component | Time | Optimization |
|-----------|------|--------------|
| Data loading | 0.2s | Regex extraction |
| Embeddings | 0.1s | Local computation |
| Relevance LLM | 1.0s | Groq inference |
| Faithfulness LLM | 1.2s | Groq inference |
| I/O | 0.1s | Minimal overhead |
| **Total** | **2.6s** | Target: <3s |

### Implemented Optimizations
- Aggressive text truncation (context: 1500 chars, response: 800 chars, query: 300 chars)
- Single turn evaluation (skip historical messages)
- Low temperature (0.1) for consistent outputs
- Regex extraction for large files (100x faster)

### Production Enhancements (Not Implemented)
- Async batch processing (5-10x throughput)
- Smart sampling (skip trivial/high-confidence responses)
- Result caching for repeated queries
- Horizontal scaling with load balancer

---

## Output Format

```json
{
  "metadata": {
    "evaluation_mode": "latest_turn_only",
    "loading_mode": "robust_dirty_extraction",
    "model_used": "llama-3.3-70b-versatile"
  },
  "evaluation_result": {
    "message_id": "msg_7",
    "query": "What are the key features...",
    "response": "The system includes...",
    "relevance_evaluation": {
      "llm_relevance_score": 9,
      "llm_completeness_score": 8,
      "semantic_similarity": 0.847
    },
    "faithfulness_evaluation": {
      "faithfulness_score": 8,
      "risk_level": "low",
      "unsupported_claims": []
    }
  },
  "performance": {
    "total_time_seconds": 2.341,
    "total_tokens": 2487,
    "total_cost_usd": 0.000249
  }
}
```

---

## Configuration

### Model Selection
```python
# evaluator.py
pipeline = LLMEvaluationPipeline(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)
```

### Truncation Limits
```python
# metrics/relevance.py
context_preview = context[:1000]
response_text = response[:800]
query_text = query[:300]
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `GROQ_API_KEY not set` | Export API key in environment |
| `No messages found` | Check logs for extraction details |
| `No AI responses found` | Verify `sender` or `role` field exists |
| High latency (>5s) | Reduce truncation limits or check Groq status |

### Debug Mode
```python
# Add at top of evaluator.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Production Checklist

- Malformed JSON handling
- Large file support (100MB+)
- Comprehensive error logging
- Async batch processing
- Smart sampling logic
- PostgreSQL storage
- Docker containerization
- Retry with exponential backoff

---

## Key Takeaways

1. **Robust data loading** prevents evaluation failures on production data
2. **Context-aware evaluation** only processes relevant turns
3. **Graduated fallbacks** balance speed and reliability
4. **Cost optimization** through Groq API and efficient sampling


---
