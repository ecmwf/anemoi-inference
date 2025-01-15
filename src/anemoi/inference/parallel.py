import os
import subprocess
import socket

import logging


def getParallelLogger(moduleName):
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    logger = logging.getLogger(moduleName)

    if global_rank != 0:
        logger.setLevel(logging.NOTSET)

    return logger

LOG = getParallelLogger(__name__)

#TODO could replace env vars with regular variables now
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
    LOG.info(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    LOG.info(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

def init_parallel(world_size):
    if world_size > 1:

        init_network()
        dist.init_process_group(
            backend="nccl",
            init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
            timeout=datetime.timedelta(minutes=3),
            world_size=world_size,
            rank=global_rank,
        )

        model_comm_group_ranks = np.arange(world_size, dtype=int)
        model_comm_group_ranks = np.arange(world_size, dtype=int)
        model_comm_group = torch.distributed.new_group(model_comm_group_ranks)
    else:
        model_comm_group = None

    return model_comm_group

def get_parallel_info():
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    global_rank = int(
        os.environ.get("SLURM_PROCID", 0)
    )  # Get rank of the current process, equivalent to dist.get_rank()
    world_size = int(os.environ.get("SLURM_NTASKS", 1))  # Total number of processes

    return global_rank, local_rank, world_size


        lead_time = to_timedelta(lead_time)
        steps = lead_time // self.checkpoint.timestep

        if global_rank == 0:
            LOG.info("World size: %d", world_size)
            LOG.info("Using autocast %s", self.autocast)
            LOG.info("Lead time: %s, time stepping: %s Forecasting %s steps", lead_time, self.checkpoint.timestep, steps)

        result = input_state.copy()  # We should not modify the input state
        result["fields"] = dict()

        start = input_state["date"]

        if world_size > 1:

            #only rank 0 logs
            if (local_rank != 0):
                LOG.handlers.clear()

            init_network()


            dist.init_process_group(
                backend="nccl",
                init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
                timeout=datetime.timedelta(minutes=3),
                world_size=world_size,
                rank=global_rank,
            )

            model_comm_group_ranks = np.arange(world_size, dtype=int)
            model_comm_group_ranks = np.arange(world_size, dtype=int)
            model_comm_group = torch.distributed.new_group(model_comm_group_ranks)
        else:
            model_comm_group = None
