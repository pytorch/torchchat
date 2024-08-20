import os
import torch
from safetensors import safe_open
from typing import Dict, Tuple, List
from collections import defaultdict
from torch._subclasses import FakeTensor


def analyze_safetensor_file(file_path: str) -> Dict[str, Tuple[torch.dtype, str]]:
    result = {}
    
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            
            # Determine dtype
            dtype = tensor.dtype
            
            # Determine tensor type
            if tensor.is_meta:
                tensor_type = "meta"
            elif isinstance(tensor, FakeTensor):
                tensor_type = "fake"
            else:
                tensor_type = "regular"
            
            result[key] = (dtype, tensor_type)
    
    return result

def analyze_safetensor_directory(directory_path: str) -> Dict[str, Dict[str, Tuple[torch.dtype, str]]]:
    all_results = {}
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.safetensors'):
            file_path = os.path.join(directory_path, filename)
            all_results[filename] = analyze_safetensor_file(file_path)
    
    return all_results

def summarize_results(all_results: Dict[str, Dict[str, Tuple[torch.dtype, str]]]) -> Dict[str, Dict[str, int]]:
    summary = {
        'dtypes': defaultdict(int),
        'tensor_types': defaultdict(int)
    }
    
    for file_results in all_results.values():
        for dtype, tensor_type in file_results.values():
            summary['dtypes'][str(dtype)] += 1
            summary['tensor_types'][tensor_type] += 1
    
    return summary

if __name__ == '__main__':
    # Example usage
    directory_path = "path/to/safetensor/directory"
    all_results = analyze_safetensor_directory(directory_path)
    summary = summarize_results(all_results)

    print("Summary of all safetensor files in the directory:")
    print("\nDtype distribution:")
    for dtype, count in summary['dtypes'].items():
        print(f"  {dtype}: {count}")

    print("\nTensor type distribution:")
    for tensor_type, count in summary['tensor_types'].items():
        print(f"  {tensor_type}: {count}")

    print("\nDetailed results for each file:")
    for filename, file_results in all_results.items():
        print(f"\nFile: {filename}")
        for tensor_name, (dtype, tensor_type) in file_results.items():
            print(f"  Tensor: {tensor_name}")
            print(f"    dtype: {dtype}")
            print(f"    type: {tensor_type}")
