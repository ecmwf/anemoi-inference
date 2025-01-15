import os
import subprocess
import socket

import logging

LOG = logging.getLogger(__name__)

def getParallelLogger():
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    logger = logging.getLogger(__name__)

    if global_rank != 0:
        logger.setLevel(logging.NOTSET)

    return logger

def init_network():
    # Get the master address from the SLURM_NODELIST environment variable
    slurm_nodelist = os.environ.get("SLURM_NODELIST")
    if not slurm_nodelist:
        raise ValueError("SLURM_NODELIST environment variable is not set.")

    # Use subprocess to execute scontrol and get the first hostname
    result = subprocess.run(
        ["scontrol", "show", "hostname", slurm_nodelist],
        stdout=subprocess.PIPE,
        text=True,
        check=True
    )
    master_addr = result.stdout.splitlines()[0]

    # Resolve the master address using nslookup
    try:
        resolved_addr = socket.gethostbyname(master_addr)
    except socket.gaierror:
        raise ValueError(f"Could not resolve hostname: {master_addr}")

    # Set the resolved address as MASTER_ADDR
    os.environ["MASTER_ADDR"] = resolved_addr

    # Calculate the MASTER_PORT using SLURM_JOBID
    slurm_jobid = os.environ.get("SLURM_JOBID")
    if not slurm_jobid:
        raise ValueError("SLURM_JOBID environment variable is not set.")

    master_port = 10000 + int(slurm_jobid[-4:])
    os.environ["MASTER_PORT"] = str(master_port)

    # Print the results for confirmation
    #print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    #print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
