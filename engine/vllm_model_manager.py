"""
vLLM Model Manager for MCS-SQL

This module provides vLLM integration for the MCS-SQL benchmark, supporting:
- Tensor parallelism across multiple GPUs (for 120B+ models)
- Structured JSON output via xgrammar/guidance backends (vLLM v0.12.0+)
- High-throughput batch generation
- Continuous batching for variable-length sequences

Usage:
    # Single vLLM instance across all GPUs (tensor parallel)
    from vllm_model_manager import vLLMModelManager

    manager = vLLMModelManager(
        model_name="openai/gpt-oss-120b",
        tensor_parallel_size=4,
        max_tokens=512,
        temperature=0.3,
    )

    # Generate with JSON schema constraint
    result = manager.generate_json(prompt, SQL_GENERATION_SCHEMA)

    # Batch generation
    results = manager.generate_json_batch(prompts, SQL_GENERATION_SCHEMA, batch_size=32)
"""

import json
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import time

try:
    from vllm import LLM, SamplingParams
    # vLLM v0.12.0+ uses StructuredOutputsParams (replaces guided_json)
    try:
        from vllm.sampling_params import StructuredOutputsParams
        HAS_STRUCTURED_OUTPUTS = True
    except ImportError:
        HAS_STRUCTURED_OUTPUTS = False
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    HAS_STRUCTURED_OUTPUTS = False
    print("Warning: vLLM not installed. Install with: pip install vllm")


@dataclass
class vLLMConfig:
    """Configuration for vLLM model."""
    model_name: str
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    max_tokens: int = 512
    temperature: float = 0.3
    gpu_memory_utilization: float = 0.95
    kv_cache_dtype: str = "auto"
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 32768
    enforce_eager: bool = False
    trust_remote_code: bool = True


class vLLMModelManager:
    """
    Manages vLLM model with tensor parallelism for high-throughput generation.
    
    Supports:
    - Single model distributed across multiple GPUs (tensor parallel)
    - Batch generation with continuous batching
    - JSON-structured output via guided decoding
    """
    
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 4,
        max_tokens: int = 512,
        temperature: float = 0.3,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 8192,
    ):
        """
        Initialize vLLM model manager.
        
        Args:
            model_name: Hugging Face model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism (4 for 4×A100)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            gpu_memory_utilization: Fraction of GPU memory to use (0.95 for 95%)
            max_model_len: Maximum sequence length (prompt + output)
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Install with: pip install vllm")
        
        self.config = vLLMConfig(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_tokens=max_tokens,
            temperature=temperature,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        print(f"\n{'='*70}")
        print(f"vLLM Model Manager Initialization")
        print(f"{'='*70}")
        print(f"  Model: {model_name}")
        print(f"  Tensor Parallel Size: {tensor_parallel_size}")
        print(f"  Max Tokens: {max_tokens}")
        print(f"  Temperature: {temperature}")
        print(f"  GPU Memory Utilization: {gpu_memory_utilization*100:.0f}%")
        print(f"  Max Model Length: {max_model_len}")
        print(f"{'='*70}\n")
        
        # Initialize vLLM model
        self.llm = self._load_model()
        
        # Create sampling params
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
        )
        
        print(f"✓ vLLM model initialized successfully")
        print(f"  Distributed across {tensor_parallel_size} GPUs")
    
    def _load_model(self) -> LLM:
        """Load vLLM model with tensor parallelism."""
        import torch
        
        # Count available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"  Available GPUs: {num_gpus}")
        
        if num_gpus < self.tensor_parallel_size:
            print(f"  Warning: Requested {self.tensor_parallel_size} GPUs, but only {num_gpus} available")
            self.tensor_parallel_size = num_gpus
        
        # Show GPU names
        for i in range(self.tensor_parallel_size):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        print(f"\n  Loading vLLM model (this may take 2-5 minutes)...")
        
        # vLLM model loading
        llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            kv_cache_dtype=self.config.kv_cache_dtype,
            enable_chunked_prefill=self.config.enable_chunked_prefill,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            enforce_eager=self.config.enforce_eager,
            trust_remote_code=self.config.trust_remote_code,
            dtype="bfloat16" if "gpt-oss" in self.model_name.lower() else "float16",
        )
        
        print(f"  Model loaded successfully")
        return llm
    
    def generate(
        self,
        prompt: str,
        stop_sequences: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate response for a single prompt.
        
        Args:
            prompt: Input prompt string
            stop_sequences: Optional list of stop sequences
            max_tokens: Optional override for max tokens
            temperature: Optional override for temperature
        
        Returns:
            Generated response string
        """
        sampling_params = SamplingParams(
            max_tokens=max_tokens if max_tokens else self.max_tokens,
            temperature=temperature if temperature else self.temperature,
            stop=stop_sequences,
            top_p=0.95,
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def generate_batch(
        self,
        prompts: List[str],
        batch_size: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        Uses vLLM's continuous batching for maximum throughput.
        
        Args:
            prompts: List of prompt strings
            batch_size: Optional batch size limit (vLLM handles this automatically if None)
            stop_sequences: Optional stop sequences
            max_tokens: Optional override for max tokens
            temperature: Optional override for temperature
            show_progress: Whether to show progress bar
        
        Returns:
            List of generated responses (same order as input prompts)
        """
        if not prompts:
            return []
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens if max_tokens else self.max_tokens,
            temperature=temperature if temperature else self.temperature,
            stop=stop_sequences,
            top_p=0.95,
        )
        
        # vLLM handles continuous batching automatically
        outputs = self.llm.generate(
            prompts,
            sampling_params,
            use_tqdm=show_progress,
        )
        
        # Extract generated text
        results = [output.outputs[0].text for output in outputs]
        return results
    
    def generate_json(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output using guided decoding.
        
        vLLM supports structured generation via:
        1. xgrammar (recommended, built-in)
        2. outlines (external, more flexible)
        
        Args:
            prompt: Input prompt string
            json_schema: JSON schema dict for structured output
            max_tokens: Optional override for max tokens
            temperature: Optional override for temperature

        Returns:
            Parsed JSON dictionary
        """
        # Create sampling params with structured outputs (vLLM v0.12.0+)
        if HAS_STRUCTURED_OUTPUTS:
            structured_outputs = StructuredOutputsParams(json=json_schema)
            sampling_params = SamplingParams(
                max_tokens=max_tokens if max_tokens else self.max_tokens,
                temperature=temperature if temperature else self.temperature,
                top_p=0.95,
                structured_outputs=structured_outputs,
            )
        else:
            # Fallback for older vLLM versions (pre-v0.12.0)
            sampling_params = SamplingParams(
                max_tokens=max_tokens if max_tokens else self.max_tokens,
                temperature=temperature if temperature else self.temperature,
                top_p=0.95,
                guided_json=json_schema,
            )

        # Generate with JSON constraint
        outputs = self.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text

        # Parse JSON response
        try:
            # Remove markdown code fences if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3].strip()

            # Extract JSON object
            start_idx = response.find("{")
            if start_idx != -1:
                end_idx = response.rfind("}") + 1
                if end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)

            # Fallback: try parsing entire response
            return json.loads(response)

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            print(f"Raw response: {response[:500]}...")
            return {"error": "JSON parsing failed", "raw_response": response}

    def generate_json_batch(
        self,
        prompts: List[str],
        json_schema: Dict[str, Any],
        batch_size: int = 32,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate structured JSON outputs for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            json_schema: JSON schema dict for all prompts
            batch_size: Batch size for processing
            max_tokens: Optional override for max tokens
            temperature: Optional override for temperature
            show_progress: Whether to show progress bar
        
        Returns:
            List of parsed JSON dictionaries
        """
        if not prompts:
            return []

        # Create sampling params with structured outputs (vLLM v0.12.0+)
        if HAS_STRUCTURED_OUTPUTS:
            structured_outputs = StructuredOutputsParams(json=json_schema)
            sampling_params = SamplingParams(
                max_tokens=max_tokens if max_tokens else self.max_tokens,
                temperature=temperature if temperature else self.temperature,
                top_p=0.95,
                structured_outputs=structured_outputs,
            )
        else:
            # Fallback for older vLLM versions (pre-v0.12.0)
            sampling_params = SamplingParams(
                max_tokens=max_tokens if max_tokens else self.max_tokens,
                temperature=temperature if temperature else self.temperature,
                top_p=0.95,
                guided_json=json_schema,
            )
        
        # Process in batches
        all_results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            outputs = self.llm.generate(
                batch_prompts,
                sampling_params,
                use_tqdm=show_progress and (i == 0),
            )
            
            # Parse JSON responses
            for output in outputs:
                response = output.outputs[0].text
                try:
                    # Remove markdown code fences
                    response = response.strip()
                    if response.startswith("```json"):
                        response = response[7:]
                    elif response.startswith("```"):
                        response = response[3:]
                    if response.endswith("```"):
                        response = response[:-3].strip()
                    
                    # Extract JSON
                    start_idx = response.find("{")
                    if start_idx != -1:
                        end_idx = response.rfind("}") + 1
                        if end_idx > start_idx:
                            json_str = response[start_idx:end_idx]
                            result = json.loads(json_str)
                        else:
                            result = json.loads(response)
                    else:
                        result = json.loads(response)
                    
                    all_results.append(result)
                
                except json.JSONDecodeError:
                    all_results.append({
                        "error": "JSON parsing failed",
                        "raw_response": response[:200]
                    })
        
        return all_results
    
    def get_gpu_memory_usage(self) -> Dict[int, Dict[str, float]]:
        """Get current GPU memory usage for all GPUs."""
        import torch
        
        memory_info = {}
        for i in range(self.tensor_parallel_size):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            memory_info[i] = {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
            }
        return memory_info
    
    def profile_memory(self) -> None:
        """Print detailed GPU memory profile."""
        print(f"\n{'='*50}")
        print(f"GPU Memory Profile")
        print(f"{'='*50}")
        
        memory_info = self.get_gpu_memory_usage()
        for gpu_id, info in memory_info.items():
            print(f"  GPU {gpu_id}:")
            print(f"    Allocated: {info['allocated_gb']:.2f} GB")
            print(f"    Reserved:  {info['reserved_gb']:.2f} GB")
        
        print(f"{'='*50}\n")


class vLLMAPIClient:
    """
    Alternative: Connect to a running vLLM API server.
    
    Use this when vLLM server is running separately:
        python -m vllm.entrypoints.api_server \\
            --model openai/gpt-oss-120b \\
            --tensor-parallel-size 4 \\
            --port 8000
    
    Then connect via:
        client = vLLMAPIClient(base_url="http://localhost:8000")
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "vllm",
    ):
        """
        Initialize vLLM API client.
        
        Args:
            base_url: vLLM API server URL
            api_key: API key (default: "vllm")
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=f"{base_url}/v1",
                api_key=api_key,
            )
        except ImportError:
            raise ImportError("OpenAI client not installed. Install with: pip install openai")
        
        self.base_url = base_url
        print(f"Connected to vLLM API server at {base_url}")
    
    def generate(
        self,
        prompt: str,
        model: str = "default",
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate response via API."""
        response = self.client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences,
        )
        return response.choices[0].text
    
    def generate_batch(
        self,
        prompts: List[str],
        model: str = "default",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> List[str]:
        """Generate responses for batch of prompts via API."""
        results = []
        for prompt in prompts:
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(response.choices[0].text)
        return results
    
    def generate_json(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        model: str = "default",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Generate JSON via API (requires server-side guided decoding support)."""
        # Note: API server may not support guided_json directly
        # Use server-side --guided-json option or parse client-side
        response = self.client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        text = response.choices[0].text
        try:
            return json.loads(text)
        except:
            return {"error": "JSON parsing failed", "raw_response": text}


# Convenience function for creating model manager
def create_vllm_manager(
    model_name: str = "openai/gpt-oss-120b",
    num_gpus: int = 4,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> vLLMModelManager:
    """
    Create vLLM model manager with specified configuration.
    
    Args:
        model_name: Hugging Face model name
        num_gpus: Number of GPUs for tensor parallelism
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Initialized vLLMModelManager
    """
    return vLLMModelManager(
        model_name=model_name,
        tensor_parallel_size=num_gpus,
        max_tokens=max_tokens,
        temperature=temperature,
        gpu_memory_utilization=0.95,
        max_model_len=8192,
    )
