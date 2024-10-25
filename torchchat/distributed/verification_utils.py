import csv
import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict
from torch._subclasses import FakeTensor
import numpy as np
from torchchat.distributed.dtensor_utils import is_dtensor, SingletonLogger
from typing import Dict, List, Tuple

logger = SingletonLogger.get_logger()


def record_module_dtypes(module):
    """Record the dtypes of all parameters and buffers in a module and return a dictionary of dtype -> list of names"""
    dtype_count = defaultdict(int)
    dtype_locations = defaultdict(list)
    fp32_locations = defaultdict(list)

    def recurse(mod, prefix=""):
        for name, param in mod.named_parameters(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            dtype = param.dtype
            dtype_count[dtype] += 1
            dtype_locations[dtype].append(full_name)
            if dtype == torch.float32:
                fp32_locations[full_name] = param

        for name, buf in mod.named_buffers(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            dtype = buf.dtype
            dtype_count[dtype] += 1
            dtype_locations[dtype].append(full_name)
            if dtype == torch.float32:
                fp32_locations[full_name] = buf

        for name, child in mod.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            recurse(child, child_prefix)

    recurse(module)
    return dtype_count, dtype_locations, fp32_locations


def find_cpu_tensors(module):
    """Find all CPU tensors in a module and return a list of their names"""
    cpu_tensors = []

    def recurse(mod):
        for name, param in mod.named_parameters(recurse=False):
            if not param.is_cuda:
                cpu_tensors.append(f"{mod.__class__.__name__}.{name}")

        for name, buf in mod.named_buffers(recurse=False):
            if not buf.is_cuda:
                cpu_tensors.append(f"{mod.__class__.__name__}.{name}")

        # Check for self.weights in RMSNorm
        if mod.__class__.__name__ == "RMSNorm" and hasattr(mod, "weights"):
            if not mod.weights.is_cuda:
                cpu_tensors.append(f"{mod.__class__.__name__}.weights")

        for name, child in mod.named_children():
            recurse(child)

    recurse(module)
    return cpu_tensors


def inspect_module_tensors(
    module: nn.Module,
) -> Dict[str, List[Tuple[str, torch.dtype]]]:
    """
    Inspect an nn.Module and return information about its tensor types.

    Args:
        module (nn.Module): The module to inspect.

    Returns:
        Dict[str, List[Tuple[str, torch.dtype]]]: A dictionary with keys 'parameters', 'buffers', and 'submodules',
        each containing a list of (name, dtype) tuples for the respective tensors.
    """
    result = defaultdict(list)

    def get_tensor_info(tensor: torch.Tensor, name: str) -> Tuple[str, str]:
        tensor_type = "Regular"
        if isinstance(tensor, FakeTensor):
            tensor_type = "Fake"
        elif tensor.is_meta:
            tensor_type = "Meta"
        return f"{name} ({tensor_type}) ({tensor.device})", str(tensor.dtype)

    # parameters
    for name, param in module.named_parameters(recurse=False):
        result["parameters"].append(get_tensor_info(param, name))

    # buffers
    for name, buffer in module.named_buffers(recurse=False):
        result["buffers"].append(get_tensor_info(buffer, name))

    # Recursively inspect submodules
    for name, submodule in module.named_children():
        submodule_info = inspect_module_tensors(submodule)
        for key, value in submodule_info.items():
            result["submodules"].extend(
                [(f"{name}.{item[0]}", item[1]) for item in value]
            )

    return dict(result)


def torch_in_fake_mode() -> bool:
    """
    Check if torch is in fake mode.
    """
    fake_mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
    return fake_mode is not None


def get_tensor_type(tensor: torch.Tensor) -> str:
    """Get the type of a tensor (regular, fake, or meta)."""
    tensor_type = "Regular"
    if isinstance(tensor, FakeTensor):
        tensor_type = "Fake"
    elif tensor.is_meta:
        tensor_type = "Meta"
    return tensor_type


def extract_and_save_weights(model, output_file):
    """
    Extract the first 4 weights from each module (including nested ones)
    and nested buffers in a model and save them to a CSV file.
    Supports regular tensors, DTensors, and nested structures.

    Args:
    model (torch.nn.Module): model to extract weights from
    output_file (str): name of the CSV file to save the results

    Returns:
    None
    """
    results = OrderedDict()

    def process_tensor(name, tensor):
        """Process a single tensor or DTensor."""
        if isinstance(tensor, torch.Tensor):
            if is_dtensor(tensor):
                tensor = tensor.full_tensor()

            if tensor.numel() >= 4:
                first_four = tensor.flatten()[:4].tolist()
            else:
                first_four = tensor.flatten().tolist() + [None] * (4 - tensor.numel())
        else:
            first_four = [str(tensor)] + [None] * 3

        return (
            [name]
            + first_four
            + [str(tensor.dtype) if hasattr(tensor, "dtype") else type(tensor).__name__]
        )

    def process_nested_buffer(name, buffer):
        """Recursively process nested buffers."""
        if isinstance(buffer, torch.Tensor):
            return [process_tensor(name, buffer)]
        elif isinstance(buffer, (list, tuple)):
            return [
                item
                for i, sub_buffer in enumerate(buffer)
                for item in process_nested_buffer(f"{name}.{i}", sub_buffer)
            ]
        elif isinstance(buffer, dict):
            return [
                item
                for key, sub_buffer in buffer.items()
                for item in process_nested_buffer(f"{name}.{key}", sub_buffer)
            ]
        else:
            return [process_tensor(name, buffer)]

    def process_module(module, prefix=""):
        """Recursively process module and its submodules."""
        module_results = OrderedDict()

        # Process parameters
        for name, param in module.named_parameters(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            module_results[full_name] = process_tensor(full_name, param.data)

        # Process buffers
        for name, buffer in module.named_buffers(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            module_results.update(
                OrderedDict(
                    [
                        (item[0], item)
                        for item in process_nested_buffer(full_name, buffer)
                    ]
                )
            )

        # Recursively process child modules
        for name, child_module in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            module_results.update(process_module(child_module, child_prefix))

        return module_results

    # Process the entire model
    results = process_module(model)

    # Write results to CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "Weight1", "Weight2", "Weight3", "Weight4", "Dtype"])
        for row in results.values():
            writer.writerow(row)

    logger.info(f"Weight information saved to {output_file}")


def compare_weight_files(file1, file2, tolerance=1e-6):
    """
    Compare two CSV files containing model weights and report any differences.

    Args:
    file1 (str): Path to the first CSV file
    file2 (str): Path to the second CSV file
    tolerance (float): Tolerance for floating point comparisons

    Returns:
    tuple: Three dictionaries containing:
           1. Entries missing in file1
           2. Entries missing in file2
           3. Weight/dtype mismatches for common entries
    """

    def load_csv(file_path):
        data = defaultdict(dict)
        with open(file_path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row["Name"]
                weights = [row[f"Weight{i}"] for i in range(1, 5)]
                dtype = row["Dtype"]
                data[name] = {"weights": weights, "dtype": dtype}
        return data

    def compare_weights(w1, w2):
        try:
            return np.allclose(
                np.array(w1, dtype=float),
                np.array(w2, dtype=float),
                rtol=tolerance,
                atol=tolerance,
                equal_nan=True,
            )
        except ValueError:
            return w1 == w2  # If conversion to float fails, compare as strings

    data1 = load_csv(file1)
    data2 = load_csv(file2)

    missing_in_file1 = {}
    missing_in_file2 = {}
    mismatches = {}

    # Check for weights in file2 but not in file1
    for name in data2:
        if name not in data1:
            missing_in_file1[name] = data2[name]

    # Check for weights in file1 but not in file2
    for name in data1:
        if name not in data2:
            missing_in_file2[name] = data1[name]

    # Compare weights and dtypes for common entries
    for name in data1:
        if name in data2:
            weights1 = data1[name]["weights"]
            weights2 = data2[name]["weights"]
            dtype1 = data1[name]["dtype"]
            dtype2 = data2[name]["dtype"]

            if not compare_weights(weights1, weights2) or dtype1 != dtype2:
                mismatches[name] = {
                    "file1": {"weights": weights1, "dtype": dtype1},
                    "file2": {"weights": weights2, "dtype": dtype2},
                }

    return missing_in_file1, missing_in_file2, mismatches


def enumerate_model_details(model, prefix="", output_file=None):
    """Prints information about the model's modules and parameters."""

    def print_info(*args):
        print(*args)
        if output_file:
            print(*args, file=output_file)

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        print_info(f"Module: {full_name}, Type: {type(module).__name__}")

        if list(module.parameters()):
            for param_name, param in module.named_parameters():
                param_type = get_tensor_type(param)
                print_info(
                    f"  Parameter: {full_name}.{param_name}, Type: {param_type}, Shape: {param.shape}, Device: {param.device}"
                )

        if list(module.buffers()):
            for buffer_name, buffer in module.named_buffers():
                buffer_type = get_tensor_type(buffer)
                print_info(
                    f"  Buffer: {full_name}.{buffer_name}, Type: {buffer_type}, Shape: {buffer.shape}, Device: {buffer.device}"
                )

        if list(module.children()):
            enumerate_model_details(module, full_name, output_file)


# compare weight files can be run directly from the command line
if __name__ == "__main__":
    file1 = "chat_master.csv"
    file2 = "chat_dist_rank0.csv"

    missing_in_file1, missing_in_file2, mismatches = compare_weight_files(file1, file2)

    print(f"Entries missing in {file1}:")
    for name, data in missing_in_file1.items():
        print(f"  {name}: {data}")

    print(f"\nEntries missing in {file2}:")
    for name, data in missing_in_file2.items():
        print(f"  {name}: {data}")

    print("\nMismatches in common entries:")
    for name, diff in mismatches.items():
        print(f"  {name}:")
        print(f"    {file1}: {diff['file1']}")
        print(f"    {file2}: {diff['file2']}")

    if not (missing_in_file1 or missing_in_file2 or mismatches):
        print("No differences found.")
