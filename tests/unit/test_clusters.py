# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from anemoi.inference.clusters import cluster_registry
from anemoi.inference.clusters import create_cluster
from anemoi.inference.clusters.distributed import DistributedCluster
from anemoi.inference.clusters.manual import ManualCluster
from anemoi.inference.clusters.mapping import EnvMapping
from anemoi.inference.clusters.mapping import MappingCluster
from anemoi.inference.clusters.mpi import MPICluster
from anemoi.inference.clusters.slurm import SlurmCluster
from anemoi.inference.context import Context


@pytest.fixture
def mock_context():
    """Create a mock context for testing."""
    context = MagicMock(spec=Context)
    context.device = MagicMock()
    context.device.type = "cuda"
    context.use_grib_paramid = False
    return context


class TestManualCluster:
    """Tests for ManualCluster."""

    def test_manual_cluster_initialization(self, mock_context):
        """Test ManualCluster initialization."""
        cluster = ManualCluster(mock_context, world_size=4, pid=0)

        assert cluster.world_size == 4
        assert cluster.global_rank == 0
        assert cluster.local_rank == 0
        assert cluster.master_addr == "localhost"
        assert isinstance(cluster.master_port, int)
        assert 10000 <= cluster.master_port < 20000

    def test_manual_cluster_invalid_world_size(self, mock_context):
        """Test ManualCluster with invalid world_size."""
        with pytest.raises(ValueError, match="world_size.*must be greater then 1"):
            ManualCluster(mock_context, world_size=0, pid=0)

        with pytest.raises(ValueError, match="world_size.*must be greater then 1"):
            ManualCluster(mock_context, world_size=-1, pid=0)

    def test_manual_cluster_different_ranks(self, mock_context):
        """Test ManualCluster with different process ranks."""
        cluster0 = ManualCluster(mock_context, world_size=4, pid=0)
        cluster1 = ManualCluster(mock_context, world_size=4, pid=1)
        cluster3 = ManualCluster(mock_context, world_size=4, pid=3)

        assert cluster0.global_rank == 0
        assert cluster1.global_rank == 1
        assert cluster3.global_rank == 3

        assert cluster0.is_master
        assert not cluster1.is_master
        assert not cluster3.is_master

    def test_manual_cluster_spawn(self, mock_context):
        """Test ManualCluster spawn functionality."""
        cluster = ManualCluster(mock_context, world_size=4, pid=0)

        mock_fn = MagicMock()

        with patch("torch.multiprocessing.Process") as mock_process:
            mock_process_instance = MagicMock()
            mock_process.return_value = mock_process_instance

            cluster.spawn(mock_fn, "arg1", "arg2")

            # Should spawn world_size - 1 processes (ranks 1, 2, 3)
            assert mock_process.call_count == 3
            assert len(cluster._spawned_processes) == 3

    def test_manual_cluster_spawn_non_master(self, mock_context):
        """Test that non-master processes don't spawn."""
        cluster = ManualCluster(mock_context, world_size=4, pid=2)

        mock_fn = MagicMock()

        with patch("torch.multiprocessing.Process") as mock_process:
            cluster.spawn(mock_fn)

            # Non-master should not spawn
            mock_process.assert_not_called()

    def test_manual_cluster_teardown(self, mock_context):
        """Test ManualCluster teardown with process cleanup."""
        cluster = ManualCluster(mock_context, world_size=4, pid=0)

        # Create mock processes
        mock_processes = [MagicMock() for _ in range(3)]
        for mock_proc in mock_processes:
            mock_proc.is_alive.return_value = False

        cluster._spawned_processes = mock_processes
        cluster._model_comm_group = None

        cluster.teardown()

        # All processes should have join called
        for mock_proc in mock_processes:
            mock_proc.is_alive.assert_called()

    def test_manual_cluster_teardown_with_alive_processes(self, mock_context):
        """Test ManualCluster teardown when processes are still alive."""
        cluster = ManualCluster(mock_context, world_size=4, pid=0)

        # Create mock processes that are alive
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345

        cluster._spawned_processes = [mock_process]
        cluster._model_comm_group = None

        cluster.teardown()

        # Should try to join, then terminate
        mock_process.join.assert_called()
        mock_process.terminate.assert_called()

    def test_manual_cluster_not_used(self):
        """Test that ManualCluster.used() returns False."""
        assert not ManualCluster.used()

    def test_manual_cluster_repr(self, mock_context):
        """Test ManualCluster string representation."""
        cluster = ManualCluster(mock_context, world_size=4, pid=2)
        repr_str = repr(cluster)

        assert "ManualCluster" in repr_str
        assert "world_size=4" in repr_str
        assert "global_rank=2" in repr_str


class TestSlurmCluster:
    """Tests for SlurmCluster."""

    def test_slurm_cluster_used_detection(self):
        """Test SlurmCluster.used() detection."""
        # Not in Slurm environment
        with patch.dict(os.environ, {}, clear=True):
            assert not SlurmCluster.used()

        # In Slurm environment
        with patch.dict(os.environ, {"SLURM_NTASKS": "4", "SLURM_JOB_NAME": "test_job"}):
            assert SlurmCluster.used()

        # Slurm but interactive shell (should not be used)
        with patch.dict(os.environ, {"SLURM_NTASKS": "4", "SLURM_JOB_NAME": "bash"}):
            assert not SlurmCluster.used()

    def test_slurm_cluster_initialization(self, mock_context):
        """Test SlurmCluster initialization."""
        with patch.dict(
            os.environ,
            {
                "SLURM_NTASKS": "8",
                "SLURM_PROCID": "3",
                "SLURM_LOCALID": "1",
                "SLURM_NODELIST": "node001",
                "SLURM_JOBID": "12345",
                "MASTER_ADDR": "192.168.1.1",
                "MASTER_PORT": "29500",
            },
        ):
            cluster = SlurmCluster(mock_context)

            assert cluster.world_size == 8
            assert cluster.global_rank == 3
            assert cluster.local_rank == 1
            assert cluster.master_addr == "192.168.1.1"
            assert cluster.master_port == 29500

    def test_slurm_cluster_master_addr_from_nodelist(self, mock_context):
        """Test SlurmCluster master_addr resolution from SLURM_NODELIST."""
        with patch.dict(
            os.environ,
            {
                "SLURM_NTASKS": "4",
                "SLURM_PROCID": "0",
                "SLURM_LOCALID": "0",
                "SLURM_NODELIST": "node[001-004]",
                "SLURM_JOBID": "12345",
            },
            clear=True,
        ):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.stdout = "node001\nnode002\nnode003\nnode004\n"

                with patch("socket.gethostbyname", return_value="192.168.1.1"):
                    cluster = SlurmCluster(mock_context)

                    assert cluster.master_addr == "192.168.1.1"
                    mock_run.assert_called_once()

    def test_slurm_cluster_master_port_from_jobid(self, mock_context):
        """Test SlurmCluster master_port generation from SLURM_JOBID."""
        with patch.dict(
            os.environ,
            {
                "SLURM_NTASKS": "4",
                "SLURM_PROCID": "0",
                "SLURM_LOCALID": "0",
                "SLURM_NODELIST": "node001",
                "SLURM_JOBID": "98765",
            },
            clear=True,
        ):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.stdout = "node001\n"

                with patch("socket.gethostbyname", return_value="192.168.1.1"):
                    cluster = SlurmCluster(mock_context)

                    # Port should be 10000 + last 4 digits of job ID
                    expected_port = 10000 + 8765
                    assert cluster.master_port == expected_port

    def test_slurm_cluster_scontrol_failure(self, mock_context):
        """Test SlurmCluster when scontrol fails."""
        with patch.dict(
            os.environ,
            {
                "SLURM_NTASKS": "4",
                "SLURM_PROCID": "0",
                "SLURM_LOCALID": "0",
                "SLURM_NODELIST": "node001",
                "SLURM_JOBID": "12345",
            },
            clear=True,
        ):
            with patch("subprocess.run", side_effect=Exception("scontrol failed")):
                with pytest.raises(Exception, match="scontrol failed"):
                    cluster = SlurmCluster(mock_context)
                    _ = cluster.master_addr


class TestMPICluster:
    """Tests for MPICluster."""

    def test_mpi_cluster_used_detection(self):
        """Test MPICluster.used() detection."""
        # Not in MPI environment
        with patch.dict(os.environ, {}, clear=True):
            assert not MPICluster.used()

        # In MPI environment (OpenMPI)
        with patch.dict(os.environ, {"OMPI_COMM_WORLD_SIZE": "4"}):
            assert MPICluster.used()

        # In MPI environment (PMI)
        with patch.dict(os.environ, {"PMI_SIZE": "4"}):
            assert MPICluster.used()

    def test_mpi_cluster_initialization(self, mock_context):
        """Test MPICluster initialization."""
        with patch.dict(
            os.environ,
            {
                "OMPI_COMM_WORLD_SIZE": "8",
                "OMPI_COMM_WORLD_RANK": "3",
                "OMPI_COMM_WORLD_LOCAL_RANK": "1",
                "MASTER_ADDR": "192.168.1.1",
                "MASTER_PORT": "29500",
            },
        ):
            cluster = MPICluster(mock_context, use_mpi_backend=True)

            assert cluster.world_size == 8
            assert cluster.global_rank == 3
            assert cluster.local_rank == 1
            assert cluster.master_addr == "192.168.1.1"
            assert cluster.master_port == 29500
            assert cluster.backend == "mpi"


class TestDistributedCluster:
    """Tests for DistributedCluster (torchrun)."""

    def test_distributed_cluster_used_detection(self):
        """Test DistributedCluster.used() detection."""
        # Not in distributed environment
        with patch.dict(os.environ, {}, clear=True):
            assert not DistributedCluster.used()

        # In distributed environment (torchrun)
        with patch.dict(os.environ, {"RANK": "3", "LOCAL_RANK": "1"}):
            assert DistributedCluster.used()

    def test_distributed_cluster_initialization(self, mock_context):
        """Test DistributedCluster initialization."""
        with patch.dict(
            os.environ,
            {"WORLD_SIZE": "8", "RANK": "3", "LOCAL_RANK": "1", "MASTER_ADDR": "192.168.1.1", "MASTER_PORT": "29500"},
        ):
            cluster = DistributedCluster(mock_context)

            assert cluster.world_size == 8
            assert cluster.global_rank == 3
            assert cluster.local_rank == 1
            assert cluster.master_addr == "192.168.1.1"
            assert cluster.master_port == 29500


class TestMappingCluster:
    """Tests for MappingCluster (custom mapping)."""

    def test_mapping_cluster_with_dict(self, mock_context):
        """Test MappingCluster with dict mapping."""
        mapping = {
            "local_rank": "MY_LOCAL_RANK",
            "global_rank": "MY_GLOBAL_RANK",
            "world_size": "MY_WORLD_SIZE",
            "master_addr": "MY_MASTER_ADDR",
            "master_port": "MY_MASTER_PORT",
            "init_method": "tcp://{master_addr}:{master_port}",
        }

        with patch.dict(
            os.environ,
            {
                "MY_WORLD_SIZE": "8",
                "MY_GLOBAL_RANK": "3",
                "MY_LOCAL_RANK": "1",
                "MY_MASTER_ADDR": "192.168.1.1",
                "MY_MASTER_PORT": "29500",
            },
        ):
            cluster = MappingCluster(mock_context, mapping=mapping)

            assert cluster.world_size == 8
            assert cluster.global_rank == 3
            assert cluster.local_rank == 1
            assert cluster.master_addr == "192.168.1.1"
            assert cluster.master_port == 29500

    def test_mapping_cluster_with_env_mapping(self, mock_context):
        """Test MappingCluster with EnvMapping object."""
        mapping = EnvMapping(
            local_rank="MY_LOCAL_RANK",
            global_rank="MY_GLOBAL_RANK",
            world_size="MY_WORLD_SIZE",
            master_addr="MY_MASTER_ADDR",
            master_port="MY_MASTER_PORT",
            init_method="env://",
        )

        with patch.dict(
            os.environ,
            {
                "MY_WORLD_SIZE": "4",
                "MY_GLOBAL_RANK": "2",
                "MY_LOCAL_RANK": "0",
                "MY_MASTER_ADDR": "localhost",
                "MY_MASTER_PORT": "12345",
            },
        ):
            cluster = MappingCluster(mock_context, mapping=mapping)

            assert cluster.world_size == 4
            assert cluster.global_rank == 2
            assert cluster.local_rank == 0
            assert cluster.init_method == "env://"

    def test_mapping_cluster_defaults(self, mock_context):
        """Test MappingCluster with missing environment variables."""
        mapping = EnvMapping(
            local_rank="MY_LOCAL_RANK",
            global_rank="MY_GLOBAL_RANK",
            world_size="MY_WORLD_SIZE",
            master_addr="MY_MASTER_ADDR",
            master_port="MY_MASTER_PORT",
        )

        with patch.dict(os.environ, {}, clear=True):
            cluster = MappingCluster(mock_context, mapping=mapping)

            # Should use defaults
            assert cluster.world_size == 1
            assert cluster.global_rank == 0
            assert cluster.local_rank == 0
            assert cluster.master_addr == ""
            assert cluster.master_port == 0

    def test_mapping_cluster_not_used(self):
        """Test that MappingCluster.used() returns False."""
        assert not MappingCluster.used()


class TestClusterRegistry:
    """Tests for cluster registry and creation."""

    def test_cluster_registry_contains_all_clusters(self):
        """Test that all cluster types are registered."""
        registered = cluster_registry.registered

        assert "manual" in registered
        assert "slurm" in registered
        assert "mpi" in registered
        assert "distributed" in registered
        assert "custom" in registered

    def test_create_cluster_with_config(self, mock_context):
        """Test create_cluster with explicit config."""
        config = {"manual": {"world_size": 4}}

        cluster = create_cluster(mock_context, config, pid=1)

        assert isinstance(cluster, ManualCluster)
        assert cluster.world_size == 4
        assert cluster.global_rank == 1

    def test_create_cluster_auto_detection(self, mock_context):
        """Test create_cluster with auto-detection."""
        with patch.dict(
            os.environ,
            {
                "SLURM_NTASKS": "4",
                "SLURM_PROCID": "0",
                "SLURM_LOCALID": "0",
                "SLURM_NODELIST": "node001",
                "SLURM_JOBID": "12345",
                "SLURM_JOB_NAME": "test_job",
                "MASTER_ADDR": "192.168.1.1",
                "MASTER_PORT": "29500",
            },
        ):
            cluster = create_cluster(mock_context, {})

            assert isinstance(cluster, SlurmCluster)
            assert cluster.world_size == 4

    def test_create_cluster_no_suitable_cluster(self, mock_context):
        """Test create_cluster when no suitable cluster found."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="No suitable cluster found"):
                create_cluster(mock_context, {})


class TestClusterBase:
    """Tests for base Cluster class functionality."""

    def test_cluster_init_method(self, mock_context):
        """Test cluster init_method property."""
        cluster = ManualCluster(mock_context, world_size=4, pid=0)

        init_method = cluster.init_method
        assert init_method.startswith("tcp://")
        assert "localhost" in init_method
        assert str(cluster.master_port) in init_method

    def test_cluster_backend_cuda(self, mock_context):
        """Test cluster backend selection for CUDA."""
        mock_context.device.type = "cuda"
        cluster = ManualCluster(mock_context, world_size=4, pid=0)

        assert cluster.backend == "nccl"

    def test_cluster_backend_cpu(self, mock_context):
        """Test cluster backend selection for CPU."""
        mock_context.device.type = "cpu"
        cluster = ManualCluster(mock_context, world_size=4, pid=0)

        assert cluster.backend == "gloo"

    def test_cluster_is_master(self, mock_context):
        """Test cluster is_master property."""
        cluster0 = ManualCluster(mock_context, world_size=4, pid=0)
        cluster1 = ManualCluster(mock_context, world_size=4, pid=1)

        assert cluster0.is_master
        assert not cluster1.is_master

    def test_cluster_address_property(self, mock_context):
        """Test cluster address property returns named tuple."""
        cluster = ManualCluster(mock_context, world_size=4, pid=0)

        address = cluster.address
        assert address.host == cluster.master_addr
        assert address.port == cluster.master_port
