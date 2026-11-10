import torch

def device_config():
    """
    Select the best available PyTorch device without interactive input.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"The device being used is the {device}, so the calculations will be sent to it")

    if device.type == "cpu":
        print("Running the script on the CPU...")

    return device


