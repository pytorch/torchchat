import csv
import torch
from collections import OrderedDict, defaultdict

import numpy as np
from distributed.dtensor_utils import is_dtensor

from distributed.logging_utils import setup_logging
logger = setup_logging(__name__)

def extract_and_save_weights(model, output_file):
    """
    Extract the first 4 weights from each module (including nested ones), buffer, 
    and nested buffer in a PyTorch model and save them to a CSV file. 
    Supports regular tensors, DTensors, and nested structures.

    Args:
    model (torch.nn.Module): The PyTorch model to extract weights from
    output_file (str): The name of the CSV file to save the results

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

        return [name] + first_four + [str(tensor.dtype) if hasattr(tensor, 'dtype') else type(tensor).__name__]

    def process_nested_buffer(name, buffer):
        """Recursively process nested buffers."""
        if isinstance(buffer, torch.Tensor):
            return [process_tensor(name, buffer)]
        elif isinstance(buffer, (list, tuple)):
            return [item for i, sub_buffer in enumerate(buffer) for item in process_nested_buffer(f"{name}.{i}", sub_buffer)]
        elif isinstance(buffer, dict):
            return [item for key, sub_buffer in buffer.items() for item in process_nested_buffer(f"{name}.{key}", sub_buffer)]
        else:
            return [process_tensor(name, buffer)]

    def process_module(module, prefix=''):
        """Recursively process module and its submodules."""
        module_results = OrderedDict()

        # Process parameters of the current module
        for name, param in module.named_parameters(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            module_results[full_name] = process_tensor(full_name, param.data)

        # Process buffers of the current module
        for name, buffer in module.named_buffers(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            module_results.update(OrderedDict([(item[0], item) for item in process_nested_buffer(full_name, buffer)]))

        # Recursively process child modules
        for name, child_module in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            module_results.update(process_module(child_module, child_prefix))

        return module_results

    # Process the entire model
    results = process_module(model)

    # Write results to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Name', 'Weight1', 'Weight2', 'Weight3', 'Weight4', 'Dtype'])
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
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['Name']
                weights = [row[f'Weight{i}'] for i in range(1, 5)]
                dtype = row['Dtype']
                data[name] = {'weights': weights, 'dtype': dtype}
        return data

    def compare_weights(w1, w2):
        try:
            return np.allclose(np.array(w1, dtype=float), np.array(w2, dtype=float), 
                               rtol=tolerance, atol=tolerance, equal_nan=True)
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
            weights1 = data1[name]['weights']
            weights2 = data2[name]['weights']
            dtype1 = data1[name]['dtype']
            dtype2 = data2[name]['dtype']

            if not compare_weights(weights1, weights2) or dtype1 != dtype2:
                mismatches[name] = {
                    'file1': {'weights': weights1, 'dtype': dtype1},
                    'file2': {'weights': weights2, 'dtype': dtype2}
                }

    return missing_in_file1, missing_in_file2, mismatches

# Can be run directly from the command line
if __name__ == "__main__":
    file1 = 'chat_master.csv'
    file2 = 'model_weights_rank1.csv'
    
    missing_in_file1, missing_in_file2, mismatches = compare_weight_files(file1, file2)
    
    '''print(f"Entries missing in {file1}:")
    for name, data in missing_in_file1.items():
        print(f"  {name}: {data}")
    
    print(f"\nEntries missing in {file2}:")
    for name, data in missing_in_file2.items():
        print(f"  {name}: {data}")
    '''
    print("\nMismatches in common entries:")
    for name, diff in mismatches.items():
        print(f"  {name}:")
        print(f"    {file1}: {diff['file1']}")
        print(f"    {file2}: {diff['file2']}")
    
    if not ( mismatches):  #missing_in_file1 or missing_in_file2 or
        print("No differences found.")
