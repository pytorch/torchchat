import csv
import torch
from collections import OrderedDict
from distributed.dtensor_utils import is_dtensor

from distributed.logging_utils import setup_logging
logger = setup_logging(__name__)

def extract_and_save_weights(model, output_file):
    """
    Extract the first 4 weights from each module and buffer in a PyTorch model
    and save them to a CSV file.

    Args:
    model (torch.nn.Module): The PyTorch model to extract weights from
    output_file (str): The name of the CSV file to save the results

    Returns:
    None
    """
    results = OrderedDict()

    # Function to process a tensor or DTensor
    def process_tensor(name, tensor):
        if isinstance(tensor, torch.Tensor):
            if is_dtensor(tensor):  # Check if it's a DTensor
                # For DTensor, we need to materialize it first
                tensor = tensor.full_tensor()
            
            if tensor.numel() >= 4:
                first_four = tensor.flatten()[:4].tolist()
            else:
                first_four = tensor.flatten().tolist() + [None] * (4 - tensor.numel())
        else:
            # Handle non-tensor types (shouldn't normally happen, but just in case)
            first_four = [str(tensor)] + [None] * 3

        return [name] + first_four + [str(tensor.dtype) if hasattr(tensor, 'dtype') else type(tensor).__name__]


    # Extract weights from named parameters
    for name, param in model.named_parameters():
        results[name] = process_tensor(name, param.data)

    # Extract values from named buffers
    for name, buffer in model.named_buffers():
        results[name] = process_tensor(name, buffer)

    # Write results to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Name', 'Weight1', 'Weight2', 'Weight3', 'Weight4', 'Dtype'])
        for row in results.values():
            writer.writerow(row)

    logger.info(f"Weight information saved to {output_file}")
