import csv
import itertools
import logging
from omegaconf import DictConfig
from pathlib import Path
from typing import Optional
import torch

from shared.CausalMCxTFGridNet import MCxTFGridNet


def get_model(
    cfg: DictConfig, ckpt_path: Optional[Path | str] = None
) -> torch.nn.Module:

    if cfg.name == "baseline":
        model = MCxTFGridNet(**cfg.params)
    else:
        raise ValueError(f"Model {cfg.name} not recognised. Add code here!")

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=get_device())
        model.load_state_dict(ckpt)

    return model


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


POSITIONS = ["pos1", "pos2", "pos3", "pos4"]


def get_session_tuples(session_file, devices, datasets, max_sessions=None):
    """Get session tuples for the specified datasets and devices.

    Args:
        session_file: Path template for session files
        devices: Device type(s) to process
        datasets: Dataset split(s) to process
        max_sessions: Optional limit on number of sessions PER DATASET (for testing)
    """
    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(devices, str):
        devices = [devices]

    sessions = []

    for ds in datasets:
        with open(session_file.format(dataset=ds), "r") as f:
            ds_sessions = list(csv.DictReader(f))

        # Limit sessions PER DATASET if max_sessions is specified
        if max_sessions is not None:
            ds_sessions = ds_sessions[:max_sessions]
            logging.info(f"Limiting {ds} to {len(ds_sessions)} sessions (max_sessions={max_sessions})")

        sessions += ds_sessions

    # Keep the total logging for reference
    if max_sessions is not None:
        logging.info(f"Total sessions across all datasets: {len(sessions)}")

    session_device_pid_tuples = []

    for device, session in itertools.product(devices, sessions):
        device_pos = "pos" + session[f"{device}_pos"]

        if device_pos not in POSITIONS:
            logging.warning(f"Device {device} not found for session {session}")
            continue

        pids = [session[pos] for pos in POSITIONS if pos != device_pos]
        for pid in pids:
            session_device_pid_tuples.append((session["session"], device, pid))

    return session_device_pid_tuples
