"""Pydantic data models for challenges, model outputs, and evaluation runs."""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class Constraint(BaseModel):
    name: str
    rule_type: str  # "forbidden_import", "forbidden_pattern", "exact_schema", "max_length"
    value: str
    description: str

class Challenge(BaseModel):
    id: str
    suite: str
    title: str
    difficulty: str  # "medium", "hard", "expert"
    category: str    # "concurrency", "refactoring", "tool_use", "reasoning", "long_context"
    language: str = "python"  # "swift", "rust", "typescript", "python"
    description: str
    prompt: str
    constraints: List[Constraint] = Field(default_factory=list)
    test_code: str  # Unit test or assertion validation code
    expected_output_pattern: Optional[str] = None

class ModelOutput(BaseModel):
    model_name: str
    raw_response: str
    thinking_content: Optional[str] = None
    code_content: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_sec: float = 0.0
    ttft_seconds: float = 0.0
    peak_memory_gb: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class VerificationResult(BaseModel):
    model_name: str
    passed_tests: int = 0
    total_tests: int = 0
    test_passed: bool = False
    test_output: str = ""
    constraint_passed: bool = True
    constraint_violations: List[str] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    error_message: Optional[str] = None

class ComparisonRecord(BaseModel):
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    challenge_id: str
    challenge_title: str
    qwen_output: ModelOutput
    qwen_verification: VerificationResult
    opus_output: Optional[ModelOutput] = None
    opus_verification: Optional[VerificationResult] = None
    winner: Optional[str] = None  # "qwen", "opus", "tie", "pending"
    notes: Optional[str] = None
