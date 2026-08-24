import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x_t = torch.tensor(x,dtype=torch.float32)
    y_t = torch.tensor(y,dtype=torch.float32)

    operations = {
        "add": torch.add,
        "multiply": torch.mul,
        "matmul": torch.matmul,
        "power": torch.pow,
        "max": torch.max
    }

    if op not in operations:
        raise ValueError(f"Invalid operation: '{op}'. Must be: {list(operations.keys())}")

    result = operations[op](x_t,y_t)
    return result.tolist()