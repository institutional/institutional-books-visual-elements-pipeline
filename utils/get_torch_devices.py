from functools import cache
import os


@cache
def get_torch_devices():
    """
    List all devices available for torch.
    Note: Does _not_ use torch, as to prevent CUDA initialization conflicts.
    """
    import json
    import subprocess, re, platform

    devices = []

    # CUDA
    try:
        # Use CUDA_VISIBLE_DEVICES if available
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)

        if cuda_visible_devices:
            devices += [f"cuda:{i}" for i in cuda_visible_devices.split(",")]

        # Use Pytorch in a separate process otherwise
        else:
            cmd = [
                "python",
                "-c",
                "import torch, json; "
                "print(json.dumps([f'cuda:{i}' for i in range(torch.cuda.device_count())]))",
            ]

            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()

            try:
                parsed = json.loads(out)
                if isinstance(parsed, list):
                    devices.extend(parsed)
            except json.JSONDecodeError:
                # In case the child prints something unexpected, ignore.
                pass
    except:
        pass

    # MPS
    if platform.system() == "Darwin":
        out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)

        if re.search(r"Apple M", out):
            devices.append("mps")

    # CPU
    devices.append("cpu")

    return devices
