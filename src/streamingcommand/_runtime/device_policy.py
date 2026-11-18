"""Generic device policy; visibility is configured only in owned child processes."""
import os
import sys

DEFAULT_DEVICE = "auto"
DEFAULT_GPU_UUID = ""
DEVICE = os.environ.get("STREAMINGCOMMAND_DEVICE", "cpu")
GPU_UUID = os.environ.get("STREAMINGCOMMAND_EXPECTED_GPU_UUID", "")
PHYSICAL_GPU = int(DEVICE.split(":")[1]) if DEVICE.startswith("cuda:") else None


def resolve_device(value="auto"):
    if value == "auto":
        import torch
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return value, ""
    if value == "mps":
        raise ValueError("MPS has not been validated for this model; select cpu explicitly")
    if not value.startswith("cuda:") or not value[5:].isdigit():
        raise ValueError("device must be auto, cpu or cuda:N")
    return value, os.environ.get("STREAMINGCOMMAND_EXPECTED_GPU_UUID", "")


def pin_device():
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if DEVICE == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE.split(":")[-1]
    return GPU_UUID
