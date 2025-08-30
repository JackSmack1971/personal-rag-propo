# Evaluation Data Formats Specification

**Document ID:** EVALUATION-DATA-FORMATS-001
**Version:** 1.0.0
**Date:** 2025-08-30
**Authors:** SPARC Specification Writer

## 1. Overview

This specification defines standardized data formats for evaluation datasets, ground truth annotations, and evaluation results in the Personal RAG Chatbot system. These formats ensure consistency across different evaluation components and enable interoperability between evaluation tools.

## 2. Core Data Structures

### 2.1 Evaluation Dataset Format

#### JSON Schema for Evaluation Datasets

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Evaluation Dataset",
  "type": "object",
  "required": ["dataset_id", "name", "version", "queries"],
  "properties": {
    "dataset_id": {
      "type": "string",
      "description": "Unique identifier for the dataset"
    },
    "name": {
      "type": "string",
      "description": "Human-readable name of the dataset"
    },
    "version": {
      "type": "string",
      "description": "Version identifier (semantic versioning)"
    },
    "description": {
      "type": "string",
      "description": "Detailed description of the dataset"
    },
    "created_at": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp of dataset creation"
    },
    "metadata": {
      "type": "object",
      "description": "Additional metadata about the dataset",
      "properties": {
        "domain": {"type": "string", "description": "Knowledge domain (e.g., 'technical', 'medical')"},
        "language": {"type": "string", "description": "Primary language (ISO 639-1)"},
        "size": {"type": "integer", "description": "Number of queries in dataset"},
        "avg_query_length": {"type": "number", "description": "Average query length in characters"},
        "source": {"type": "string", "description": "Source of the dataset"}
      }
    },
    "queries": {
      "type": "array",
      "description": "List of evaluation queries",
      "items": {"$ref": "#/$defs/EvaluationQuery"}
    },
    "documents": {
      "type": "array",
      "description": "List of source documents (optional)",
      "items": {"$ref": "#/$defs/SourceDocument"}
    }
  },
  "$defs": {
    "EvaluationQuery": {
      "type": "object",
      "required": ["query_id", "query_text", "relevant_documents"],
      "properties": {
        "query_id": {"type": "string", "description": "Unique query identifier"},
        "query_text": {"type": "string", "description": "The query text"},
        "query_type": {
          "type": "string",
          "enum": ["factual", "analytical", "comparative", "temporal"],
          "description": "Type of query for analysis"
        },
        "difficulty": {
          "type": "string",
          "enum": ["easy", "medium", "hard"],
          "description": "Difficulty level of the query"
        },
        "relevant_documents": {
          "type": "array",
          "description": "List of relevant document IDs",
          "items": {"type": "string"}
        },
        "ground_truth_answer": {
          "type": "string",
          "description": "Expected answer text (optional)"
        },
        "citations": {
          "type": "array",
          "description": "Expected citations with spans",
          "items": {"$ref": "#/$defs/CitationSpan"}
        },
        "metadata": {
          "type": "object",
          "description": "Query-specific metadata"
        }
      }
    },
    "SourceDocument": {
      "type": "object",
      "required": ["document_id", "content"],
      "properties": {
        "document_id": {"type": "string", "description": "Unique document identifier"},
        "title": {"type": "string", "description": "Document title"},
        "content": {"type": "string", "description": "Full document text"},
        "file_path": {"type": "string", "description": "Original file path"},
        "metadata": {
          "type": "object",
          "description": "Document metadata (author, date, etc.)"
        }
      }
    },
    "CitationSpan": {
      "type": "object",
      "required": ["file_name", "start_char", "end_char"],
      "properties": {
        "file_name": {"type": "string", "description": "Source file name"},
        "page_number": {"type": "integer", "description": "Page number (optional)"},
        "start_char": {"type": "integer", "description": "Start character position"},
        "end_char": {"type": "integer", "description": "End character position"},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        "context_text": {"type": "string", "description": "Surrounding context"}
      }
    }
  }
}
```

#### Example Evaluation Dataset

```json
{
  "dataset_id": "personal_rag_eval_v1.0.0",
  "name": "Personal RAG Evaluation Dataset",
  "version": "1.0.0",
  "description": "Comprehensive evaluation dataset for personal knowledge retrieval",
  "created_at": "2025-08-30T12:00:00Z",
  "metadata": {
    "domain": "personal",
    "language": "en",
    "size": 500,
    "avg_query_length": 45,
    "source": "synthetic"
  },
  "queries": [
    {
      "query_id": "query_001",
      "query_text": "What are the key features of the new project management tool?",
      "query_type": "factual",
      "difficulty": "medium",
      "relevant_documents": ["doc_001", "doc_002"],
      "ground_truth_answer": "The new project management tool includes features such as task tracking, team collaboration, time management, and reporting capabilities.",
      "citations": [
        {
          "file_name": "project_tool_guide.pdf",
          "page_number": 3,
          "start_char": 1250,
          "end_char": 1350,
          "confidence_score": 0.95,
          "context_text": "Key features include task tracking and team collaboration..."
        }
      ],
      "metadata": {
        "expected_answer_length": 150,
        "num_relevant_docs": 2
      }
    }
  ],
  "documents": [
    {
      "document_id": "doc_001",
      "title": "Project Management Tool Overview",
      "content": "The new project management tool provides comprehensive features for team collaboration and task tracking...",
      "file_path": "docs/project_tool.pdf",
      "metadata": {
        "author": "Product Team",
        "created_date": "2025-08-01",
        "word_count": 1250
      }
    }
  ]
}
```

### 2.2 Ground Truth Annotations Format

#### Citation Annotation Format

```json
{
  "annotation_id": "citation_gt_001",
  "query_id": "query_001",
  "annotator_id": "annotator_001",
  "timestamp": "2025-08-30T12:00:00Z",
  "citations": [
    {
      "citation_id": "cit_001",
      "file_name": "project_tool_guide.pdf",
      "page_number": 3,
      "start_char": 1250,
      "end_char": 1350,
      "text": "Key features include task tracking and team collaboration",
      "relevance_score": 1.0,
      "annotation_confidence": 0.9,
      "justification": "Directly answers the query about key features"
    }
  ],
  "quality_checks": {
    "span_accuracy": "verified",
    "text_extraction": "successful",
    "relevance_assessment": "high_confidence"
  },
  "metadata": {
    "annotation_time_seconds": 120,
    "tools_used": ["pdf_viewer", "text_selector"],
    "notes": "Clear and unambiguous citation span"
  }
}
```

#### Answer Quality Annotation Format

```json
{
  "annotation_id": "answer_quality_gt_001",
  "query_id": "query_001",
  "annotator_id": "annotator_002",
  "timestamp": "2025-08-30T12:30:00Z",
  "answer_text": "The new project management tool includes features such as task tracking, team collaboration, time management, and reporting capabilities.",
  "quality_scores": {
    "completeness": 0.9,
    "accuracy": 0.95,
    "relevance": 0.9,
    "clarity": 0.85,
    "conciseness": 0.8
  },
  "overall_score": 0.88,
  "aspect_ratings": {
    "factual_correctness": "fully_correct",
    "comprehensive_coverage": "mostly_comprehensive",
    "answer_structure": "well_structured",
    "language_quality": "good"
  },
  "feedback": {
    "strengths": ["Comprehensive feature list", "Clear language"],
    "weaknesses": ["Could mention specific tool capabilities"],
    "suggestions": ["Add examples of each feature"]
  },
  "metadata": {
    "annotation_time_seconds": 180,
    "expertise_level": "domain_expert",
    "review_status": "approved"
  }
}
```

### 2.3 Evaluation Results Format

#### Comprehensive Evaluation Results

```json
{
  "evaluation_id": "eval_run_20250830_001",
  "timestamp": "2025-08-30T13:00:00Z",
  "system_version": "personal-rag-v2.1.0",
  "dataset_id": "personal_rag_eval_v1.0.0",
  "configuration": {
    "model": "gpt-4",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "retrieval_k": 5,
    "reranking_enabled": true
  },
  "summary": {
    "total_queries": 500,
    "successful_queries": 495,
    "failed_queries": 5,
    "average_latency_ms": 1250,
    "total_evaluation_time_seconds": 675
  },
  "retrieval_metrics": {
    "hit@1": {"value": 0.72, "confidence_interval": [0.68, 0.76]},
    "hit@3": {"value": 0.85, "confidence_interval": [0.82, 0.88]},
    "hit@5": {"value": 0.91, "confidence_interval": [0.89, 0.93]},
    "ndcg@5": {"value": 0.78, "confidence_interval": [0.75, 0.81]},
    "mrr": {"value": 0.76, "confidence_interval": [0.73, 0.79]},
    "map@5": {"value": 0.74, "confidence_interval": [0.71, 0.77]}
  },
  "citation_metrics": {
    "span_accuracy": {
      "mean_exact_jaccard": 0.85,
      "mean_tolerance_jaccard": 0.92,
      "perfect_matches": 380,
      "good_matches": 445
    },
    "citation_completeness": {
      "value": 0.88,
      "total_claims": 1250,
      "cited_claims": 1100
    },
    "citation_correctness": {
      "value": 0.91,
      "total_citations": 850,
      "correct_citations": 773
    }
  },
  "answer_quality_metrics": {
    "factual_consistency": {
      "value": 0.89,
      "mean_consistency": 0.87,
      "consistency_scores": [0.9, 0.85, 0.92, ...]
    },
    "hallucination_rate": {
      "value": 0.06,
      "total_claims": 1250,
      "hallucinated_claims": 75
    }
  },
  "performance_metrics": {
    "latency_stats": {
      "mean": 1250,
      "p50": 1100,
      "p95": 2100,
      "p99": 3200,
      "min": 800,
      "max": 4500
    },
    "resource_usage": {
      "memory_mb": {
        "mean": 1850,
        "peak": 2450,
        "std": 150
      },
      "cpu_percent": {
        "mean": 45,
        "peak": 78,
        "std": 8
      }
    },
    "api_usage": {
      "total_tokens": 125000,
      "total_requests": 500,
      "estimated_cost_usd": 2.50
    }
  },
  "query_results": [
    {
      "query_id": "query_001",
      "query_text": "What are the key features of the new project management tool?",
      "latency_ms": 1150,
      "retrieval_results": {
        "retrieved_docs": ["doc_001", "doc_002", "doc_003", "doc_004", "doc_005"],
        "relevance_scores": [0.92, 0.88, 0.76, 0.65, 0.54],
        "ground_truth_relevant": ["doc_001", "doc_002", "doc_004"]
      },
      "generated_answer": "The new project management tool includes features such as task tracking, team collaboration, time management, and reporting capabilities.",
      "citations": [
        {
          "file_name": "project_tool_guide.pdf",
          "page_number": 3,
          "start_char": 1250,
          "end_char": 1350,
          "confidence_score": 0.95
        }
      ],
      "individual_metrics": {
        "hit@3": 1.0,
        "ndcg@3": 0.85,
        "citation_span_accuracy": 0.92,
        "factual_consistency": 0.9
      }
    }
  ],
  "statistical_analysis": {
    "significance_tests": {
      "retrieval_improvement": {
        "baseline_mean": 0.75,
        "current_mean": 0.78,
        "p_value": 0.02,
        "significant": true,
        "effect_size": 0.3
      }
    },
    "confidence_intervals": {
      "all_metrics": "calculated",
      "confidence_level": 0.95
    }
  },
  "metadata": {
    "evaluation_environment": {
      "hardware": "Intel i7-9750H, 16GB RAM",
      "software": "Python 3.11, PyTorch 2.8.0",
      "system_load": "low"
    },
    "evaluation_parameters": {
      "timeout_seconds": 30,
      "max_retries": 3,
      "random_seed": 42
    },
    "quality_assurance": {
      "data_validation": "passed",
      "metric_calculation": "verified",
      "statistical_tests": "passed"
    }
  }
}
```

## 3. Data Format Specifications

### 3.1 File Formats and Extensions

| Data Type | Primary Format | Alternative Formats | File Extension |
|-----------|----------------|-------------------|----------------|
| Evaluation Dataset | JSON | YAML, CSV | `.eval.json` |
| Ground Truth | JSON | YAML | `.gt.json` |
| Evaluation Results | JSON | YAML, CSV | `.results.json` |
| Benchmark Data | JSON | Parquet, HDF5 | `.benchmark.json` |
| Configuration | YAML | JSON | `.config.yaml` |

### 3.2 Directory Structure

```
data/
├── evaluation/
│   ├── datasets/
│   │   ├── personal_rag_eval_v1.0.0.eval.json
│   │   ├── domain_specific_eval_v1.1.0.eval.json
│   │   └── synthetic_queries_v2.0.0.eval.json
│   ├── ground_truth/
│   │   ├── personal_rag_eval_v1.0.0.gt.json
│   │   └── annotations/
│   │       ├── annotator_001/
│   │       └── annotator_002/
│   └── results/
│       ├── eval_run_20250830_001.results.json
│       ├── ab_test_moe_vs_baseline_001.results.json
│       └── benchmark_weekly_20250830.results.json
├── benchmarks/
│   ├── performance/
│   │   ├── latency_benchmarks.json
│   │   └── throughput_benchmarks.json
│   └── quality/
│       ├── retrieval_quality_benchmarks.json
│       └── citation_accuracy_benchmarks.json
└── config/
    ├── evaluation_defaults.config.yaml
    ├── ab_testing_config.config.yaml
    └── performance_benchmark_config.config.yaml
```

### 3.3 Data Validation

#### JSON Schema Validation

```python
import jsonschema
from typing import Dict, Any, List

class DataFormatValidator:
    """Validates evaluation data formats against schemas"""

    def __init__(self):
        self.schemas = self._load_schemas()

    def validate_evaluation_dataset(self, data: Dict[str, Any]) -> List[str]:
        """Validate evaluation dataset format"""
        return self._validate_data(data, self.schemas['evaluation_dataset'])

    def validate_ground_truth(self, data: Dict[str, Any]) -> List[str]:
        """Validate ground truth format"""
        return self._validate_data(data, self.schemas['ground_truth'])

    def validate_evaluation_results(self, data: Dict[str, Any]) -> List[str]:
        """Validate evaluation results format"""
        return self._validate_data(data, self.schemas['evaluation_results'])

    def _validate_data(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Generic data validation"""
        errors = []

        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e.message}")

        # Additional custom validations
        errors.extend(self._custom_validations(data))

        return errors

    def _custom_validations(self, data: Dict[str, Any]) -> List[str]:
        """Custom validation rules beyond JSON schema"""
        errors = []

        # Validate query IDs are unique
        if 'queries' in data:
            query_ids = [q.get('query_id') for q in data['queries']]
            if len(query_ids) != len(set(query_ids)):
                errors.append("Duplicate query IDs found")

        # Validate citation spans are valid
        if 'citations' in data:
            for citation in data['citations']:
                if citation.get('start_char', 0) >= citation.get('end_char', 0):
                    errors.append(f"Invalid citation span: start >= end")

        return errors

    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load JSON schemas for validation"""
        return {
            'evaluation_dataset': {
                "type": "object",
                "required": ["dataset_id", "queries"],
                "properties": {
                    "dataset_id": {"type": "string"},
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["query_id", "query_text", "relevant_documents"]
                        }
                    }
                }
            },
            'ground_truth': {
                "type": "object",
                "required": ["annotation_id", "query_id"],
                "properties": {
                    "annotation_id": {"type": "string"},
                    "query_id": {"type": "string"}
                }
            },
            'evaluation_results': {
                "type": "object",
                "required": ["evaluation_id", "retrieval_metrics"],
                "properties": {
                    "evaluation_id": {"type": "string"},
                    "retrieval_metrics": {"type": "object"}
                }
            }
        }
```

### 3.4 Data Compression and Storage

#### Compression Formats

```python
import gzip
import json
from typing import Dict, Any

class DataCompressor:
    """Handles compression and decompression of evaluation data"""

    @staticmethod
    def compress_evaluation_data(data: Dict[str, Any], compression_level: int = 6) -> bytes:
        """Compress evaluation data using gzip"""
        json_str = json.dumps(data, indent=None, separators=(',', ':'))
        return gzip.compress(json_str.encode('utf-8'), compresslevel=compression_level)

    @staticmethod
    def decompress_evaluation_data(compressed_data: bytes) -> Dict[str, Any]:
        """Decompress evaluation data"""
        json_str = gzip.decompress(compressed_data).decode('utf-8')
        return json.loads(json_str)

    @staticmethod
    def estimate_compression_ratio(original_data: Dict[str, Any]) -> float:
        """Estimate compression ratio for data"""
        original_size = len(json.dumps(original_data).encode('utf-8'))
        compressed_size = len(DataCompressor.compress_evaluation_data(original_data))

        return original_size / compressed_size if compressed_size > 0 else 1.0
```

## 4. Data Loading and Processing

### 4.1 Dataset Loader

```python
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

class EvaluationDatasetLoader:
    """Loads and processes evaluation datasets"""

    def __init__(self, data_directory: str = "data/evaluation"):
        self.data_directory = Path(data_directory)

    def load_dataset(self, dataset_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Load evaluation dataset by ID"""

        if version:
            filename = f"{dataset_id}_{version}.eval.json"
        else:
            # Find latest version
            pattern = f"{dataset_id}_*.eval.json"
            matching_files = list(self.data_directory.glob(f"datasets/{pattern}"))

            if not matching_files:
                raise FileNotFoundError(f"No dataset found for ID: {dataset_id}")

            # Sort by version (assuming semantic versioning)
            matching_files.sort(key=lambda x: x.stem.split('_')[-1], reverse=True)
            filename = matching_files[0].name

        file_path = self.data_directory / "datasets" / filename

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate data format
        validator = DataFormatValidator()
        errors = validator.validate_evaluation_dataset(data)

        if errors:
            raise ValueError(f"Dataset validation errors: {errors}")

        return data

    def load_ground_truth(self, dataset_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Load ground truth annotations"""

        if version:
            filename = f"{dataset_id}_{version}.gt.json"
        else:
            pattern = f"{dataset_id}_*.gt.json"
            matching_files = list(self.data_directory.glob(f"ground_truth/{pattern}"))

            if not matching_files:
                raise FileNotFoundError(f"No ground truth found for dataset: {dataset_id}")

            matching_files.sort(key=lambda x: x.stem.split('_')[-1], reverse=True)
            filename = matching_files[0].name

        file_path = self.data_directory / "ground_truth" / filename

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """Get list of available evaluation datasets"""

        datasets = []

        for file_path in self.data_directory.glob("datasets/*.eval.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                datasets.append({
                    'dataset_id': data.get('dataset_id'),
                    'name': data.get('name'),
                    'version': data.get('version'),
                    'description': data.get('description'),
                    'size': len(data.get('queries', [])),
                    'file_path': str(file_path)
                })

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading dataset {file_path}: {e}")
                continue

        return datasets
```

### 4.2 Results Storage

```python
import json
from datetime import datetime
from typing import Dict, Any

class EvaluationResultsStorage:
    """Handles storage and retrieval of evaluation results"""

    def __init__(self, results_directory: str = "data/evaluation/results"):
        self.results_directory = Path(results_directory)
        self.results_directory.mkdir(parents=True, exist_ok=True)

    def save_results(self, results: Dict[str, Any], evaluation_type: str = "standard") -> str:
        """Save evaluation results to file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_id = results.get('evaluation_id', f"eval_{timestamp}")

        filename = f"{evaluation_id}.results.json"
        file_path = self.results_directory / filename

        # Add metadata
        results['_metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'file_path': str(file_path),
            'evaluation_type': evaluation_type
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return str(file_path)

    def load_results(self, evaluation_id: str) -> Dict[str, Any]:
        """Load evaluation results by ID"""

        pattern = f"{evaluation_id}.results.json"
        matching_files = list(self.results_directory.glob(pattern))

        if not matching_files:
            raise FileNotFoundError(f"No results found for evaluation: {evaluation_id}")

        with open(matching_files[0], 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_results(self, evaluation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available evaluation results"""

        results = []

        for file_path in self.results_directory.glob("*.results.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                result_info = {
                    'evaluation_id': data.get('evaluation_id'),
                    'timestamp': data.get('timestamp'),
                    'system_version': data.get('system_version'),
                    'dataset_id': data.get('dataset_id'),
                    'file_path': str(file_path)
                }

                if evaluation_type is None or data.get('_metadata', {}).get('evaluation_type') == evaluation_type:
                    results.append(result_info)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading results {file_path}: {e}")
                continue

        return results

    def compare_results(self, evaluation_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple evaluation results"""

        results_data = {}
        for eval_id in evaluation_ids:
            results_data[eval_id] = self.load_results(eval_id)

        # Extract key metrics for comparison
        comparison = {
            'evaluations': evaluation_ids,
            'metrics_comparison': {},
            'performance_comparison': {}
        }

        # Compare retrieval metrics
        retrieval_metrics = ['hit@1', 'hit@3', 'hit@5', 'ndcg@5', 'mrr']

        for metric in retrieval_metrics:
            metric_values = {}
            for eval_id, data in results_data.items():
                if 'retrieval_metrics' in data and metric in data['retrieval_metrics']:
                    metric_values[eval_id] = data['retrieval_metrics'][metric]['value']

            if metric_values:
                comparison['metrics_comparison'][metric] = metric_values

        # Compare performance metrics
        perf_metrics = ['latency_stats', 'resource_usage']

        for metric in perf_metrics:
            perf_values = {}
            for eval_id, data in results_data.items():
                if 'performance_metrics' in data and metric in data['performance_metrics']:
                    perf_values[eval_id] = data['performance_metrics'][metric]

            if perf_values:
                comparison['performance_comparison'][metric] = perf_values

        return comparison
```

## 5. Data Quality Assurance

### 5.1 Data Integrity Checks

```python
class DataQualityChecker:
    """Performs quality assurance checks on evaluation data"""

    def __init__(self):
        self.checks = [
            self._check_query_uniqueness,
            self._check_document_references,
            self._check_citation_validity,
            self._check_ground_truth_consistency
        ]

    def run_quality_checks(self, dataset: Dict[str, Any]) -> List[str]:
        """Run all quality checks on dataset"""

        issues = []

        for check_func in self.checks:
            try:
                check_issues = check_func(dataset)
                issues.extend(check_issues)
            except Exception as e:
                issues.append(f"Check failed: {check_func.__name__}: {e}")

        return issues

    def _check_query_uniqueness(self, dataset: Dict[str, Any]) -> List[str]:
        """Check that all query IDs are unique"""

        issues = []
        queries = dataset.get('queries', [])
        query_ids = [q.get('query_id') for q in queries]

        duplicates = set([x for x in query_ids if query_ids.count(x) > 1])

        if duplicates:
            issues.append(f"Duplicate query IDs found: {duplicates}")

        return issues

    def _check_document_references(self, dataset: Dict[str, Any]) -> List[str]:
        """Check that all referenced documents exist"""

        issues = []
        queries = dataset.get('queries', [])
        documents = dataset.get('documents', [])
        doc_ids = set(d.get('document_id') for d in documents)

        for query in queries:
            relevant_docs = query.get('relevant_documents', [])
            for doc_id in relevant_docs:
                if doc_id not in doc_ids:
                    issues.append(f"Query {query['query_id']} references missing document: {doc_id}")

        return issues

    def _check_citation_validity(self, dataset: Dict[str, Any]) -> List[str]:
        """Check citation span validity"""

        issues = []
        queries = dataset.get('queries', [])

        for query in queries:
            citations = query.get('citations', [])
            for citation in citations:
                start = citation.get('start_char', 0)
                end = citation.get('end_char', 0)

                if start >= end:
                    issues.append(f"Invalid citation span in query {query['query_id']}: start >= end")

                if start < 0 or end < 0:
                    issues.append(f"Negative citation positions in query {query['query_id']}")

        return issues

    def _check_ground_truth_consistency(self, dataset: Dict[str, Any]) -> List[str]:
        """Check consistency between different ground truth elements"""

        issues = []
        queries = dataset.get('queries', [])

        for query in queries:
            relevant_docs = set(query.get('relevant_documents', []))
            citations = query.get('citations', [])

            # Check that cited documents are in relevant documents
            cited_docs = set()
            for citation in citations:
                file_name = citation.get('file_name')
                if file_name:
                    # Convert file name to document ID (assuming naming convention)
                    doc_id = file_name.replace('.pdf', '').replace('.txt', '').replace('.md', '')
                    cited_docs.add(doc_id)

            uncited_relevant = relevant_docs - cited_docs
            if uncited_relevant:
                issues.append(f"Query {query['query_id']} has relevant documents not cited: {uncited_relevant}")

        return issues
```

## 6. Usage Examples

### 6.1 Creating an Evaluation Dataset

```python
from pathlib import Path
import json

def create_sample_evaluation_dataset():
    """Create a sample evaluation dataset"""

    dataset = {
        "dataset_id": "sample_eval_v1.0.0",
        "name": "Sample Evaluation Dataset",
        "version": "1.0.0",
        "description": "Sample dataset for testing evaluation framework",
        "created_at": "2025-08-30T12:00:00Z",
        "metadata": {
            "domain": "general",
            "language": "en",
            "size": 2
        },
        "queries": [
            {
                "query_id": "sample_query_001",
                "query_text": "What is machine learning?",
                "query_type": "factual",
                "difficulty": "easy",
                "relevant_documents": ["doc_ml_intro"],
                "ground_truth_answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
                "citations": [
                    {
                        "file_name": "ml_guide.pdf",
                        "page_number": 1,
                        "start_char": 100,
                        "end_char": 200,
                        "confidence_score": 0.95
                    }
                ]
            }
        ],
        "documents": [
            {
                "document_id": "doc_ml_intro",
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence...",
                "file_path": "docs/ml_intro.pdf"
            }
        ]
    }

    # Save dataset
    output_path = Path("data/evaluation/datasets/sample_eval_v1.0.0.eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return str(output_path)
```

### 6.2 Loading and Using Evaluation Data

```python
def run_evaluation_example():
    """Example of loading and using evaluation data"""

    # Load dataset
    loader = EvaluationDatasetLoader()
    dataset = loader.load_dataset("sample_eval_v1.0.0")

    print(f"Loaded dataset with {len(dataset['queries'])} queries")

    # Load ground truth
    ground_truth = loader.load_ground_truth("sample_eval_v1.0.0")

    # Run evaluation
    evaluator = CitationEvaluator({})

    for query in dataset['queries']:
        # Simulate system response
        simulated_answer = query['ground_truth_answer']  # Perfect answer for demo
        simulated_citations = query['citations']

        # Evaluate
        results = evaluator.evaluate_answer_citations(
            simulated_answer,
            simulated_citations,
            {doc['document_id']: doc['content'] for doc in dataset['documents']},
            query
        )

        print(f"Query {query['query_id']}: Citation accuracy = {results['overall_scores']['overall_citation_accuracy']:.3f}")

    # Save results
    results_storage = EvaluationResultsStorage()
    results_path = results_storage.save_results({
        'evaluation_id': 'sample_eval_001',
        'timestamp': '2025-08-30T12:00:00Z',
        'dataset_id': 'sample_eval_v1.0.0',
        'results': []  # Would contain actual results
    })

    print(f"Results saved to: {results_path}")
```

This specification provides comprehensive data format standards for evaluation datasets, ensuring consistency, interoperability, and quality across the Personal RAG Chatbot evaluation ecosystem.