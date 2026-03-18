"""
Error logging utility for CUDA OOM and other GPU errors.
Saves error details to errors/ directory for debugging.
"""

import os
import json
import traceback
import torch
from datetime import datetime
from typing import Optional, Dict, Any


class ErrorLogger:
    """Logger for capturing and saving GPU/CUDA errors."""
    
    def __init__(self, base_output_dir: str = "outputs/benchmark_results"):
        self.errors_dir = os.path.join(base_output_dir, "errors")
        os.makedirs(self.errors_dir, exist_ok=True)
        
    def log_cuda_error(self, error: Exception, context: Dict[str, Any]) -> str:
        """
        Log CUDA-related error with full context.
        
        Args:
            error: The exception that was raised
            context: Dictionary with context info (question_id, gpu_id, batch_size, etc.)
            
        Returns:
            Path to the saved error file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_type = type(error).__name__
        
        # Gather GPU memory info
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "cuda_available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device() if torch.cuda.is_available() else -1,
            }
            
            # Get memory info for each GPU
            for i in range(torch.cuda.device_count()):
                try:
                    gpu_info[f"gpu_{i}"] = {
                        "name": torch.cuda.get_device_name(i),
                        "allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
                        "reserved_gb": torch.cuda.memory_reserved(i) / 1e9,
                        "max_allocated_gb": torch.cuda.max_memory_allocated(i) / 1e9,
                        "free_gb": (torch.cuda.get_device_properties(i).total_memory - 
                                   torch.cuda.memory_allocated(i)) / 1e9,
                    }
                except Exception:
                    gpu_info[f"gpu_{i}"] = {"error": "Could not query GPU"}
        
        # Build error report
        error_report = {
            "timestamp": timestamp,
            "error_type": error_type,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context,
            "gpu_info": gpu_info,
            "pytorch_version": torch.__version__,
        }
        
        # Save to file
        filename = f"{timestamp}_{error_type}_{context.get('question_id', 'unknown')}.json"
        filepath = os.path.join(self.errors_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(error_report, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"ERROR LOGGED: {filepath}")
        print(f"Error Type: {error_type}")
        print(f"Message: {str(error)}")
        if gpu_info:
            print(f"GPU Memory Status:")
            for key, val in gpu_info.items():
                if key.startswith("gpu_") and isinstance(val, dict) and "allocated_gb" in val:
                    print(f"  {key}: Allocated={val['allocated_gb']:.2f}GB, "
                          f"Reserved={val['reserved_gb']:.2f}GB, "
                          f"Free={val['free_gb']:.2f}GB")
        print(f"{'='*80}\n")
        
        return filepath
    
    def log_oom_offload(self, context: Dict[str, Any], offloaded_items: list) -> str:
        """
        Log when CPU offloading was triggered.
        
        Args:
            context: Context information
            offloaded_items: List of items that were offloaded to CPU
            
        Returns:
            Path to log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_entry = {
            "timestamp": timestamp,
            "event": "CPU_OFFLOAD_TRIGGERED",
            "context": context,
            "offloaded_items": offloaded_items,
        }
        
        filename = f"{timestamp}_offload_{context.get('question_id', 'unknown')}.json"
        filepath = os.path.join(self.errors_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, default=str)
        
        return filepath


def check_gpu_memory(threshold_gb: float = 5.0) -> tuple:
    """
    Check if GPU memory is running low.
    
    Args:
        threshold_gb: Minimum free memory threshold in GB
        
    Returns:
        (is_low, free_gb, allocated_gb)
    """
    if not torch.cuda.is_available():
        return False, 0, 0
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
    allocated = torch.cuda.memory_allocated(device) / 1e9
    free = total_memory - allocated
    
    is_low = free < threshold_gb
    return is_low, free, allocated


def clear_gpu_memory(verbose: bool = True):
    """
    Aggressively clear GPU memory.
    
    Args:
        verbose: Whether to print memory status
    """
    import gc
    
    if not torch.cuda.is_available():
        return
    
    # Python garbage collection
    gc.collect()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    
    if verbose:
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"  [Memory Cleared] Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")


class MemoryManager:
    """
    Manages GPU memory with CPU offloading capability.
    """
    
    def __init__(self, threshold_gb: float = 5.0, errors_dir: str = None):
        """
        Initialize memory manager.
        
        Args:
            threshold_gb: Free memory threshold to trigger offloading
            errors_dir: Directory for error logs
        """
        self.threshold_gb = threshold_gb
        self.error_logger = ErrorLogger(errors_dir) if errors_dir else None
        self.cpu_cache = {}  # Store tensors on CPU when needed
        
    def check_and_offload(self, tensors_to_offload: Dict[str, torch.Tensor], 
                         context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Check GPU memory and offload tensors to CPU if needed.
        
        Args:
            tensors_to_offload: Dictionary of tensors that can be offloaded
            context: Context information for logging
            
        Returns:
            Dictionary with tensors moved to CPU (if offloaded)
        """
        is_low, free_gb, allocated_gb = check_gpu_memory(self.threshold_gb)
        
        if is_low:
            print(f"\n[MEMORY WARNING] GPU memory low: {free_gb:.2f}GB free")
            print(f"  Triggering CPU offload...")
            
            # Move tensors to CPU
            offloaded = {}
            for name, tensor in tensors_to_offload.items():
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    offloaded[name] = tensor.cpu()
                    print(f"    Offloaded {name}: {tensor.shape} to CPU")
            
            # Clear GPU memory
            clear_gpu_memory(verbose=False)
            
            # Log the offload event
            if self.error_logger:
                self.error_logger.log_oom_offload(context, list(offloaded.keys()))
            
            return offloaded
        
        return tensors_to_offload
    
    def restore_to_gpu(self, offloaded_tensors: Dict[str, torch.Tensor], 
                      device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        Restore tensors from CPU back to GPU.
        
        Args:
            offloaded_tensors: Tensors on CPU
            device: Target GPU device
            
        Returns:
            Tensors moved back to GPU
        """
        if device is None:
            device = torch.device('cuda')
        
        restored = {}
        for name, tensor in offloaded_tensors.items():
            if isinstance(tensor, torch.Tensor) and not tensor.is_cuda:
                restored[name] = tensor.to(device)
            else:
                restored[name] = tensor
        
        return restored
