# Citation Accuracy Evaluation Specification

**Document ID:** CITATION-ACCURACY-001
**Version:** 1.0.0
**Date:** 2025-08-30
**Authors:** SPARC Specification Writer

## 1. Overview

This specification defines comprehensive metrics and methodologies for evaluating citation accuracy in the Personal RAG Chatbot system. Citation accuracy is critical for maintaining trust and verifiability of AI-generated answers, ensuring that all claims can be traced back to their source documents with precise character-level spans.

## 2. Citation Types and Formats

### 2.1 Supported Citation Formats

The system supports multiple citation formats to accommodate different use cases and user preferences:

#### Standard Format
```
[file:page:start-end]
```

**Example:**
```
[annual_report.pdf:5:1234-1289]
```

#### Extended Format (with context)
```
[file:page:start-end | excerpt: "context text..."]
```

#### Multi-span Format
```
[file:page:start1-end1,start2-end2]
```

#### Cross-reference Format
```
[file1:page1:start1-end1] [file2:page2:start2-end2]
```

### 2.2 Citation Metadata

Each citation contains the following metadata:

```python
@dataclass
class CitationSpan:
    """Represents a single citation span"""
    file_name: str
    page_number: int
    start_char: int
    end_char: int
    confidence_score: float = 1.0
    extraction_method: str = "llm"  # llm, regex, hybrid
    context_text: str = ""
    relevance_score: float = 1.0
```

## 3. Citation Accuracy Metrics

### 3.1 Span-Level Accuracy Metrics

#### Character-Level Precision and Recall

**Definition**: Measures the exactness of character spans in citations.

**Character Precision**:
```
Character_Precision = |True_Chars ∩ Predicted_Chars| / |Predicted_Chars|
```

**Character Recall**:
```
Character_Recall = |True_Chars ∩ Predicted_Chars| / |True_Chars|
```

**Character F1-Score**:
```
Character_F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Where:
- `True_Chars`: Set of characters in ground truth span
- `Predicted_Chars`: Set of characters in predicted span

#### Token-Level Accuracy

**Definition**: Measures accuracy at the token/word level, more forgiving than character-level.

**Token Precision**:
```
Token_Precision = |True_Tokens ∩ Predicted_Tokens| / |Predicted_Tokens|
```

**Token Recall**:
```
Token_Recall = |True_Tokens ∩ Predicted_Tokens| / |True_Tokens|
```

#### Overlap Coefficients

**Jaccard Similarity**:
```
Jaccard = |True_Chars ∩ Predicted_Chars| / |True_Chars ∪ Predicted_Chars|
```

**Sørensen-Dice Coefficient**:
```
Dice = 2 × |True_Chars ∩ Predicted_Chars| / (|True_Chars| + |Predicted_Chars|)
```

### 3.2 Citation-Level Accuracy Metrics

#### Citation Completeness

**Definition**: Measures whether all relevant information in the answer is properly cited.

```
Citation_Completeness = Cited_Claims / Total_Claims
```

Where:
- `Cited_Claims`: Number of claims in answer that have citations
- `Total_Claims`: Total number of factual claims in answer

#### Citation Correctness

**Definition**: Measures whether citations actually support the claims they cite.

```
Citation_Correctness = Correct_Citations / Total_Citations
```

Where:
- `Correct_Citations`: Citations that actually support their associated claims
- `Total_Citations`: Total number of citations in answer

#### Citation Relevance

**Definition**: Measures whether cited spans are relevant to the query and answer.

```
Citation_Relevance = Relevant_Citations / Total_Citations
```

### 3.3 Answer-Level Accuracy Metrics

#### Factual Consistency

**Definition**: Measures whether the answer is factually consistent with cited sources.

```
Factual_Consistency = Consistent_Claims / Total_Claims
```

#### Hallucination Detection

**Definition**: Identifies claims in answers that are not supported by any source.

```
Hallucination_Rate = Unsupported_Claims / Total_Claims
```

#### Citation Density

**Definition**: Measures the appropriate level of citation coverage.

```
Citation_Density = Total_Citation_Chars / Answer_Chars
```

## 4. Evaluation Methodology

### 4.1 Ground Truth Creation

#### Manual Annotation Process

1. **Query Generation**: Create diverse queries representing different information needs
2. **Answer Generation**: Generate answers using the system
3. **Span Identification**: Human annotators identify relevant character spans in source documents
4. **Citation Validation**: Verify that citations correctly support answer claims

#### Annotation Guidelines

**Span Selection Criteria:**
- Select minimal span containing all information needed to support the claim
- Prefer contiguous text over fragmented spans
- Include sufficient context for comprehension
- Avoid including irrelevant information

**Quality Control:**
- Inter-annotator agreement > 0.85 (character-level Jaccard)
- Blind re-annotation of 10% of samples
- Regular calibration sessions for annotators

### 4.2 Automated Evaluation

#### Span Matching Algorithm

```python
def evaluate_citation_spans(true_spans: List[CitationSpan],
                          predicted_spans: List[CitationSpan],
                          tolerance_chars: int = 10) -> Dict[str, float]:
    """
    Evaluate citation span accuracy with tolerance for minor variations.

    Args:
        true_spans: Ground truth citation spans
        predicted_spans: Predicted citation spans
        tolerance_chars: Character tolerance for matching

    Returns:
        Dictionary of accuracy metrics
    """

    # Create character-level sets for each span
    true_char_sets = [set(range(s.start_char, s.end_char + 1)) for s in true_spans]
    pred_char_sets = [set(range(s.start_char, s.end_char + 1)) for s in predicted_spans]

    # Calculate pairwise similarities
    similarities = []
    for true_set in true_char_sets:
        for pred_set in pred_char_sets:
            # Exact match
            exact_overlap = len(true_set & pred_set)
            exact_union = len(true_set | pred_set)
            exact_jaccard = exact_overlap / exact_union if exact_union > 0 else 0

            # Tolerance match (expand predicted spans)
            expanded_pred = set()
            for char_pos in pred_set:
                expanded_pred.update(range(
                    max(0, char_pos - tolerance_chars),
                    char_pos + tolerance_chars + 1
                ))

            tolerance_overlap = len(true_set & expanded_pred)
            tolerance_union = len(true_set | expanded_pred)
            tolerance_jaccard = tolerance_overlap / tolerance_union if tolerance_union > 0 else 0

            similarities.append({
                'exact_jaccard': exact_jaccard,
                'tolerance_jaccard': tolerance_jaccard,
                'overlap_chars': exact_overlap
            })

    # Calculate aggregate metrics
    if similarities:
        best_matches = [max(sims, key=lambda x: x['tolerance_jaccard']) for sims in [similarities]]

        return {
            'mean_exact_jaccard': sum(s['exact_jaccard'] for s in best_matches) / len(best_matches),
            'mean_tolerance_jaccard': sum(s['tolerance_jaccard'] for s in best_matches) / len(best_matches),
            'perfect_matches': sum(1 for s in best_matches if s['exact_jaccard'] == 1.0),
            'good_matches': sum(1 for s in best_matches if s['tolerance_jaccard'] >= 0.8)
        }

    return {'mean_exact_jaccard': 0.0, 'mean_tolerance_jaccard': 0.0, 'perfect_matches': 0, 'good_matches': 0}
```

#### Citation Validation Algorithm

```python
def validate_citation_support(answer_text: str,
                           citations: List[CitationSpan],
                           source_documents: Dict[str, str]) -> Dict[str, float]:
    """
    Validate that citations actually support the claims they reference.

    Args:
        answer_text: Generated answer text
        citations: List of citation spans
        source_documents: Dictionary of document contents

    Returns:
        Dictionary of validation metrics
    """

    validation_results = []

    for citation in citations:
        # Extract cited text
        doc_content = source_documents.get(citation.file_name, "")
        if not doc_content:
            validation_results.append({'valid': False, 'reason': 'document_not_found'})
            continue

        try:
            cited_text = doc_content[citation.start_char:citation.end_char]
        except IndexError:
            validation_results.append({'valid': False, 'reason': 'span_out_of_bounds'})
            continue

        # Find the claim this citation supports (simplified)
        # In practice, this would use NLP to match claims to citations
        claim_text = extract_associated_claim(answer_text, citation)

        # Validate semantic support
        support_score = calculate_semantic_support(claim_text, cited_text)

        validation_results.append({
            'valid': support_score >= 0.7,  # Threshold for validity
            'support_score': support_score,
            'cited_length': len(cited_text),
            'claim_length': len(claim_text) if claim_text else 0
        })

    # Calculate aggregate metrics
    valid_citations = sum(1 for r in validation_results if r['valid'])
    total_citations = len(validation_results)

    return {
        'citation_validity_rate': valid_citations / total_citations if total_citations > 0 else 0,
        'mean_support_score': sum(r['support_score'] for r in validation_results) / total_citations if total_citations > 0 else 0,
        'invalid_citations': total_citations - valid_citations,
        'validation_details': validation_results
    }
```

### 4.3 LLM-Based Evaluation

#### Semantic Support Calculation

```python
def calculate_semantic_support(claim: str, cited_text: str) -> float:
    """
    Calculate how well the cited text supports the claim using LLM evaluation.

    Args:
        claim: The claim being made
        cited_text: The cited supporting text

    Returns:
        Support score between 0.0 and 1.0
    """

    evaluation_prompt = f"""
    Evaluate how well the following cited text supports the given claim.

    Claim: {claim}

    Cited Text: {cited_text}

    Rate the support on a scale of 0.0 to 1.0 where:
    - 1.0 = The cited text fully and directly supports the claim
    - 0.8 = The cited text strongly supports the claim
    - 0.6 = The cited text moderately supports the claim
    - 0.4 = The cited text weakly supports the claim
    - 0.2 = The cited text tangentially relates to the claim
    - 0.0 = The cited text does not support the claim at all

    Consider:
    - Direct factual support
    - Logical entailment
    - Contextual relevance
    - Absence of contradictory information

    Provide only the numerical score:
    """

    # Call LLM for evaluation
    response = call_llm_for_evaluation(evaluation_prompt)

    try:
        score = float(response.strip())
        return max(0.0, min(1.0, score))  # Clamp to [0,1]
    except ValueError:
        return 0.5  # Default to neutral score on parsing error
```

## 5. Implementation Architecture

### 5.1 Core Modules

#### CitationEvaluator (`src/eval/citation_evaluator.py`)

```python
class CitationEvaluator:
    """Comprehensive citation accuracy evaluation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_evaluator = LLMCitationValidator(config.get('llm_config', {}))

    def evaluate_answer_citations(self,
                                answer: str,
                                citations: List[CitationSpan],
                                source_documents: Dict[str, str],
                                ground_truth: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of citation accuracy for an answer.

        Args:
            answer: Generated answer text
            citations: List of citation spans
            source_documents: Source document contents
            ground_truth: Optional ground truth for comparison

        Returns:
            Dictionary of evaluation metrics
        """

        results = {
            'span_accuracy': {},
            'citation_quality': {},
            'answer_consistency': {},
            'hallucination_detection': {}
        }

        # Evaluate span accuracy
        if ground_truth and 'citation_spans' in ground_truth:
            results['span_accuracy'] = self._evaluate_span_accuracy(
                citations, ground_truth['citation_spans']
            )

        # Evaluate citation quality
        results['citation_quality'] = self._evaluate_citation_quality(
            answer, citations, source_documents
        )

        # Evaluate answer consistency
        results['answer_consistency'] = self._evaluate_answer_consistency(
            answer, citations, source_documents
        )

        # Detect hallucinations
        results['hallucination_detection'] = self._detect_hallucinations(
            answer, citations, source_documents
        )

        # Calculate overall scores
        results['overall_scores'] = self._calculate_overall_scores(results)

        return results

    def _evaluate_span_accuracy(self, predicted_spans: List[CitationSpan],
                              true_spans: List[CitationSpan]) -> Dict[str, float]:
        """Evaluate accuracy of citation spans"""
        return evaluate_citation_spans(true_spans, predicted_spans)

    def _evaluate_citation_quality(self, answer: str, citations: List[CitationSpan],
                                 source_documents: Dict[str, str]) -> Dict[str, float]:
        """Evaluate overall quality of citations"""
        return validate_citation_support(answer, citations, source_documents)

    def _evaluate_answer_consistency(self, answer: str, citations: List[CitationSpan],
                                   source_documents: Dict[str, str]) -> Dict[str, float]:
        """Evaluate factual consistency of answer with citations"""

        consistency_results = []

        for citation in citations:
            doc_content = source_documents.get(citation.file_name, "")
            if not doc_content:
                continue

            cited_text = self._extract_cited_text(citation, doc_content)
            claim = self._extract_associated_claim(answer, citation)

            if claim and cited_text:
                consistency_score = self.llm_evaluator.evaluate_consistency(claim, cited_text)
                consistency_results.append(consistency_score)

        return {
            'mean_consistency': sum(consistency_results) / len(consistency_results) if consistency_results else 0.0,
            'consistency_variance': statistics.variance(consistency_results) if len(consistency_results) > 1 else 0.0,
            'consistent_citations': sum(1 for s in consistency_results if s >= 0.8)
        }

    def _detect_hallucinations(self, answer: str, citations: List[CitationSpan],
                             source_documents: Dict[str, str]) -> Dict[str, Any]:
        """Detect potential hallucinations in the answer"""

        # Extract claims from answer
        claims = self._extract_claims_from_answer(answer)

        hallucination_results = []

        for claim in claims:
            # Check if claim is supported by any citation
            supported = False
            supporting_citations = []

            for citation in citations:
                doc_content = source_documents.get(citation.file_name, "")
                if not doc_content:
                    continue

                cited_text = self._extract_cited_text(citation, doc_content)
                support_score = self.llm_evaluator.evaluate_support(claim, cited_text)

                if support_score >= 0.7:
                    supported = True
                    supporting_citations.append({
                        'citation': citation,
                        'support_score': support_score
                    })

            hallucination_results.append({
                'claim': claim,
                'supported': supported,
                'supporting_citations': supporting_citations
            })

        unsupported_claims = [r for r in hallucination_results if not r['supported']]

        return {
            'hallucination_rate': len(unsupported_claims) / len(claims) if claims else 0.0,
            'unsupported_claims': unsupported_claims,
            'total_claims': len(claims)
        }

    def _calculate_overall_scores(self, evaluation_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall citation accuracy scores"""

        scores = {}

        # Span accuracy score
        span_acc = evaluation_results.get('span_accuracy', {})
        scores['span_accuracy_score'] = span_acc.get('mean_tolerance_jaccard', 0.0)

        # Citation quality score
        cite_qual = evaluation_results.get('citation_quality', {})
        scores['citation_quality_score'] = cite_qual.get('citation_validity_rate', 0.0)

        # Answer consistency score
        ans_cons = evaluation_results.get('answer_consistency', {})
        scores['answer_consistency_score'] = ans_cons.get('mean_consistency', 0.0)

        # Hallucination score (inverse of hallucination rate)
        hall_det = evaluation_results.get('hallucination_detection', {})
        hallucination_rate = hall_det.get('hallucination_rate', 1.0)
        scores['hallucination_score'] = 1.0 - hallucination_rate

        # Overall citation accuracy (weighted average)
        weights = {
            'span_accuracy_score': 0.3,
            'citation_quality_score': 0.3,
            'answer_consistency_score': 0.25,
            'hallucination_score': 0.15
        }

        scores['overall_citation_accuracy'] = sum(
            scores[metric] * weight for metric, weight in weights.items()
        )

        return scores
```

#### LLM Citation Validator (`src/eval/llm_citation_validator.py`)

```python
class LLMCitationValidator:
    """LLM-based citation validation and evaluation"""

    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.client = OpenRouterClient(llm_config)

    def evaluate_support(self, claim: str, cited_text: str) -> float:
        """Evaluate how well cited text supports a claim"""
        return calculate_semantic_support(claim, cited_text)

    def evaluate_consistency(self, claim: str, cited_text: str) -> float:
        """Evaluate factual consistency between claim and cited text"""
        # Similar to support but focuses on factual accuracy
        return self.evaluate_support(claim, cited_text)

    def detect_hallucination(self, claim: str, available_sources: List[str]) -> float:
        """Detect if a claim is hallucinated given available sources"""

        hallucination_prompt = f"""
        Given the following claim and available source texts, determine if the claim is supported by the sources.

        Claim: {claim}

        Available Sources:
        {chr(10).join(f"- {source[:200]}..." for source in available_sources)}

        Is this claim:
        1. Fully supported by the sources
        2. Partially supported by the sources
        3. Not supported (hallucinated)

        Provide only the number (1, 2, or 3):
        """

        response = self.client.call(hallucination_prompt)

        try:
            result = int(response.strip())
            # Convert to confidence score (1.0 = supported, 0.0 = hallucinated)
            return {1: 1.0, 2: 0.5, 3: 0.0}.get(result, 0.5)
        except ValueError:
            return 0.5
```

## 6. Configuration and Parameters

### 6.1 Evaluation Configuration

```yaml
citation_accuracy:
  enabled: true

  # Span matching parameters
  span_matching:
    tolerance_chars: 10
    min_overlap_threshold: 0.5
    max_span_length: 10000

  # LLM evaluation parameters
  llm_evaluation:
    model: "openrouter/auto"
    temperature: 0.0
    max_tokens: 100
    retry_attempts: 3

  # Quality thresholds
  quality_thresholds:
    span_accuracy: 0.8
    citation_validity: 0.7
    answer_consistency: 0.8
    hallucination_tolerance: 0.1

  # Performance settings
  performance:
    batch_size: 10
    cache_results: true
    parallel_evaluation: true

  # Reporting settings
  reporting:
    detailed_span_analysis: true
    hallucination_details: true
    confidence_intervals: true
```

### 6.2 Quality Control Parameters

```yaml
quality_control:
  inter_annotator_agreement:
    target_agreement: 0.85
    min_annotators: 2
    recalibration_threshold: 0.8

  validation_checks:
    span_bounds_check: true
    document_existence_check: true
    text_extraction_validation: true

  error_handling:
    max_retries: 3
    fallback_scoring: 0.5
    error_logging: true
```

## 7. Test Cases and Validation

### 7.1 Unit Test Cases

#### Span Accuracy Tests

```python
def test_span_accuracy():
    """Test citation span accuracy calculation"""

    # Perfect match
    true_span = CitationSpan("doc1.pdf", 1, 100, 150)
    pred_span = CitationSpan("doc1.pdf", 1, 100, 150)

    accuracy = evaluate_citation_spans([true_span], [pred_span])
    assert accuracy['mean_exact_jaccard'] == 1.0
    assert accuracy['perfect_matches'] == 1

    # Partial overlap
    true_span = CitationSpan("doc1.pdf", 1, 100, 200)
    pred_span = CitationSpan("doc1.pdf", 1, 150, 250)  # 50 char overlap

    accuracy = evaluate_citation_spans([true_span], [pred_span])
    expected_jaccard = 50 / 200  # |A∩B| / |A∪B| = 50 / 200 = 0.25
    assert abs(accuracy['mean_exact_jaccard'] - expected_jaccard) < 0.01

    # No overlap
    true_span = CitationSpan("doc1.pdf", 1, 100, 150)
    pred_span = CitationSpan("doc1.pdf", 1, 200, 250)

    accuracy = evaluate_citation_spans([true_span], [pred_span])
    assert accuracy['mean_exact_jaccard'] == 0.0
```

#### Citation Validation Tests

```python
def test_citation_validation():
    """Test citation support validation"""

    answer = "The company reported $10M in revenue."
    citations = [CitationSpan("report.pdf", 1, 500, 550)]

    source_docs = {
        "report.pdf": "... The company achieved $10M in total revenue for Q1 ..."
    }

    validation = validate_citation_support(answer, citations, source_docs)

    # Should have high validity rate
    assert validation['citation_validity_rate'] > 0.8
    assert validation['mean_support_score'] > 0.7
```

### 7.2 Integration Test Cases

#### End-to-End Citation Evaluation

```python
def test_end_to_end_citation_evaluation():
    """Test complete citation evaluation pipeline"""

    # Sample query and answer
    query = "What was the company's Q1 revenue?"
    answer = "The company reported $10M in Q1 revenue. [annual_report.pdf:5:1234-1289]"

    # Ground truth
    ground_truth = {
        'citation_spans': [
            CitationSpan("annual_report.pdf", 5, 1230, 1290)  # Close but not exact
        ]
    }

    # Source document
    source_docs = {
        "annual_report.pdf": "..." * 1000 + "The company reported $10M in Q1 revenue from product sales." + "..." * 1000
    }

    # Extract citations from answer
    citations = extract_citations_from_answer(answer)

    # Evaluate
    evaluator = CitationEvaluator({})
    results = evaluator.evaluate_answer_citations(
        answer, citations, source_docs, ground_truth
    )

    # Verify results structure
    assert 'span_accuracy' in results
    assert 'citation_quality' in results
    assert 'overall_scores' in results

    # Check overall score is reasonable
    overall_score = results['overall_scores']['overall_citation_accuracy']
    assert 0.0 <= overall_score <= 1.0
```

## 8. Performance Characteristics

### 8.1 Latency Requirements

| Operation | Target Latency | Acceptable Range |
|-----------|----------------|------------------|
| Span Matching | <5ms | <20ms |
| Citation Validation | <100ms | <500ms |
| LLM Evaluation | <2s | <10s |
| Complete Evaluation | <3s | <15s |

### 8.2 Resource Requirements

| Resource | Baseline Usage | Peak Usage |
|----------|----------------|------------|
| Memory | <200MB | <1GB |
| CPU | <20% | <50% |
| API Calls | <5 per evaluation | <20 per evaluation |

### 8.3 Scalability Targets

- **Concurrent Evaluations**: Support 50+ simultaneous evaluations
- **Evaluation Throughput**: Process 100+ queries/hour
- **Result Storage**: Maintain 30+ days of evaluation history
- **Cache Hit Rate**: >80% for repeated evaluations

## 9. Integration with Evaluation Harness

### 9.1 Metric Registration

```python
# Register citation metrics with evaluation harness
citation_metrics = {
    'citation_span_accuracy': CitationSpanAccuracy(),
    'citation_completeness': CitationCompleteness(),
    'citation_correctness': CitationCorrectness(),
    'factual_consistency': FactualConsistency(),
    'hallucination_rate': HallucinationRate()
}

evaluation_harness.register_metrics(citation_metrics)
```

### 9.2 Automated Evaluation Pipeline

```python
class AutomatedCitationEvaluator:
    """Automated citation evaluation for continuous monitoring"""

    def __init__(self, config):
        self.config = config
        self.evaluator = CitationEvaluator(config)
        self.alerts = CitationAlertSystem(config)

    def evaluate_query_batch(self, queries: List[Dict[str, Any]]):
        """Evaluate citation accuracy for a batch of queries"""

        results = []

        for query in queries:
            try:
                evaluation = self.evaluator.evaluate_answer_citations(
                    query['answer'],
                    query['citations'],
                    query['source_documents'],
                    query.get('ground_truth')
                )

                results.append(evaluation)

                # Check for alerts
                self.alerts.check_thresholds(evaluation, query)

            except Exception as e:
                logger.error(f"Citation evaluation failed for query {query['id']}: {e}")
                results.append({'error': str(e)})

        return results

    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive citation evaluation report"""

        # Aggregate results
        aggregated = self._aggregate_results(results)

        # Generate visualizations
        visualizations = self._generate_visualizations(aggregated)

        # Identify trends and issues
        analysis = self._analyze_trends(aggregated)

        return {
            'summary': aggregated,
            'visualizations': visualizations,
            'analysis': analysis,
            'recommendations': self._generate_recommendations(analysis)
        }
```

## 10. Future Enhancements

### 10.1 Advanced Features

- **Multi-Modal Citations**: Support for image and table citations
- **Temporal Citation Tracking**: Citation accuracy over time
- **User Feedback Integration**: Human validation of citation quality
- **Automated Ground Truth Generation**: LLM-assisted annotation

### 10.2 Machine Learning Enhancements

- **Citation Quality Prediction**: ML model to predict citation accuracy
- **Hallucination Detection**: Advanced ML-based hallucination detection
- **Citation Recommendation**: Suggest optimal citations for answers
- **Adaptive Evaluation**: Learn from user feedback to improve evaluation

### 10.3 Integration Extensions

- **External Validation Services**: Integration with third-party fact-checking
- **Cross-System Citation**: Citations across multiple document collections
- **Semantic Citation Matching**: Meaning-based rather than exact span matching
- **Citation Network Analysis**: Analyze citation relationships and patterns

---

This specification provides a comprehensive framework for evaluating citation accuracy, ensuring that the Personal RAG Chatbot maintains high standards of verifiability and trustworthiness in its answers.