import torch

def device_config():
    """
    Configuring the hardware and software ecosystem.
    Detecting the potential presence of a graphics card to use it for computations


    If an NVIDIA graphics card or an integrated graphics card such as Intel ARC Graphics or Apple Silicon is detected


    If computations are sent to the CPU, user confirmation will be required to confirm that they are ready
    to send everything to it 

 

    --------------------------------------------------------------------------------------------------------
    To use an XPU (Intel Arc Graphics), install the IPEX version of PyTorch (intel_extension_for_pytorch).

    If this is not installed, the model will run on the CPU.

    If an XPU is required, Python 3.11 must be installed on the machine 
    --------------------------------------------------------------------------------------------------------



    --------------------------------------------------------------------------------------------------------
    To use ROCm (Radeon Open Compute) on an integrated AMD graphics card, install version rocm6.1

    During recognition, if an AMD graphics card is detected, the system will switch to CUDA; an AMD chip is recognised

    in exactly the same way as an NVIDIA graphics card 
    ---------------------------------------------------------------------------------------------------------
    """ 

    #check if the machine has an NVIDIA GPU with CUDA technology
    if torch.cuda.is_available():
        device = torch.device("cuda") #NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps") #Apple Silicon GPU
    elif torch.backends.xpu.is_available():
        device = torch.device("xpu") #Intel  GPU
    else:
        device = torch.device("cpu") #CPU
    print(f"The device being used is the {device}, so the calculations will be sent to it")

    #Loop requests a halt if the calculations are being processed on the CPU
    if device == torch.device("cpu"): 
        while True:
            choix = input(" The calculations are about to be sent to your CPU. Do you want to continue or cancel? (Yes: Y/No: N) \n")
            if choix == "Y" or choix == "y":
                print("Running the script on the CPU...")
                break
            elif choix == "N" or choix == "n":
                print("Script stopped...")
                exit()
            else:
                print("Please enter a valid entry (Yes: Y, No: N)")

