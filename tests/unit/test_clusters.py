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
from anemoi.inference.clusters.manual import ManualClient
from anemoi.inference.clusters.manual import ManualSpawner
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


class TestManualSpawner:
    """Tests for ManualSpawner."""

    def test_manual_spawner_initialization(self):
        """Test ManualSpawner initialization."""
        # ManualSpawner is decorated with @main_argument("world_size")
        # So world_size can be passed as the second positional arg (after context)
        spawner = ManualSpawner(4)

        assert spawner._world_size == 4
        assert spawner._spawned_processes == []

    def test_manual_spawner_invalid_world_size(self):
        """Test ManualSpawner with invalid world_size."""
        with pytest.raises(ValueError, match="world_size must be at least 1"):
            ManualSpawner(0)

        with pytest.raises(ValueError, match="world_size must be at least 1"):
            ManualSpawner(-1)

    def test_manual_spawner_spawn(self):
        """Test ManualSpawner spawn functionality."""
        spawner = ManualSpawner(4, port=12345)

        mock_fn = MagicMock()
        mock_config = MagicMock()

        with patch("torch.multiprocessing.Process") as mock_process:
            mock_process_instance = MagicMock()
            mock_process.return_value = mock_process_instance

            spawner.spawn(mock_fn, mock_config)

            # Should spawn world_size processes (ranks 0, 1, 2, 3)
            assert mock_process.call_count == 4
            assert len(spawner._spawned_processes) == 4

    def test_manual_spawner_teardown(self):
        """Test ManualSpawner teardown with process cleanup."""
        # Ensure environment marker is not set
        with patch.dict(os.environ, {}, clear=True):
            spawner = ManualSpawner(4)

            # Create mock processes
            mock_processes = [MagicMock() for _ in range(4)]
            for mock_proc in mock_processes:
                mock_proc.is_alive.return_value = False

            spawner._spawned_processes = mock_processes

            spawner.teardown()

            # All processes should have is_alive checked
            for mock_proc in mock_processes:
                mock_proc.is_alive.assert_called()

    def test_manual_spawner_teardown_with_alive_processes(self):
        """Test ManualSpawner teardown when processes are still alive."""
        # Ensure environment marker is not set
        with patch.dict(os.environ, {}, clear=True):
            spawner = ManualSpawner(4)

            # Create mock processes that are alive
            mock_process = MagicMock()
            mock_process.is_alive.return_value = True
            mock_process.pid = 12345

            spawner._spawned_processes = [mock_process]

            spawner.teardown()

            # Should try to join, then terminate
            mock_process.join.assert_called()
            mock_process.terminate.assert_called()

    def test_manual_spawner_not_used(self):
        """Test that ManualSpawner.used() returns False."""
        assert not ManualSpawner.used()


class TestManualClient:
    """Tests for ManualClient."""

    def test_manual_client_initialization(self):
        """Test ManualClient initialization."""
        client = ManualClient(
            world_size=4,
            local_rank=0,
            global_rank=0,
            master_addr="localhost",
            master_port=12345,
        )

        assert client.world_size == 4
        assert client.global_rank == 0
        assert client.local_rank == 0
        assert client.master_addr == "localhost"
        assert client.master_port == 12345

    def test_manual_client_different_ranks(self):
        """Test ManualClient with different process ranks."""
        client = ManualClient(
            world_size=4,
            local_rank=1,
            global_rank=1,
            master_addr="localhost",
            master_port=12345,
        )

        assert client.global_rank == 1
        assert not client.is_master

    def test_manual_client_used(self):
        """Test ManualClient.used() detection."""
        # ManualClient is now always available (returns True) since it's explicitly instantiated
        assert ManualClient.used()

    def test_manual_client_repr(self):
        """Test ManualClient string representation."""
        client = ManualClient(
            world_size=4,
            local_rank=2,
            global_rank=2,
            master_addr="localhost",
            master_port=12345,
        )
        repr_str = repr(client)

        assert "ManualClient" in repr_str
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
        with patch.dict(os.environ, {"SLURM_NTASKS": "4", "SLURM_JOB_NAME": "test_job", "SLURM_LOCALID": "1"}):
            assert SlurmCluster.used()

        # Slurm but interactive shell (should not be used)
        with patch.dict(os.environ, {"SLURM_NTASKS": "4", "SLURM_JOB_NAME": "bash"}):
            assert not SlurmCluster.used()

    def test_slurm_cluster_initialization(self):
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
            cluster = SlurmCluster()

            assert cluster.world_size == 8
            assert cluster.global_rank == 3
            assert cluster.local_rank == 1
            assert cluster.master_addr == "192.168.1.1"
            assert cluster.master_port == 29500

    def test_slurm_cluster_master_addr_from_nodelist(self):
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
                    cluster = SlurmCluster()

                    assert cluster.master_addr == "192.168.1.1"
                    mock_run.assert_called_once()

    def test_slurm_cluster_master_port_from_jobid(self):
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
                    cluster = SlurmCluster()

                    # Port should be 10000 + last 4 digits of job ID
                    expected_port = 10000 + 8765
                    assert cluster.master_port == expected_port

    def test_slurm_cluster_scontrol_failure(self):
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
                    cluster = SlurmCluster()
                    _ = cluster.master_addr


class TestMPICluster:
    """Tests for MPICluster."""

    def test_mpi_cluster_used_detection(self):
        """Test MPICluster.used() detection."""
        # Not in MPI environment
        with patch.dict(os.environ, {}, clear=True):
            assert not MPICluster.used()

        with patch.dict(os.environ, {"MASTER_ADDR": "localhost", "MASTER_PORT": "29500"}):
            # In MPI environment (OpenMPI)
            with patch.dict(
                os.environ,
                {"OMPI_COMM_WORLD_SIZE": "4", "OMPI_COMM_WORLD_RANK": "0", "OMPI_COMM_WORLD_LOCAL_RANK": "0"},
            ):
                assert MPICluster.used()

            # In MPI environment (PMI)
            with patch.dict(os.environ, {"PMI_SIZE": "4", "PMI_RANK": "0"}):
                assert MPICluster.used()

    def test_mpi_cluster_initialization(self):
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
            cluster = MPICluster(use_mpi_backend=True)

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
        with patch.dict(
            os.environ,
            {"RANK": "3", "WORLD_SIZE": "4", "LOCAL_RANK": "1", "MASTER_ADDR": "localhost", "MASTER_PORT": "29500"},
        ):
            assert DistributedCluster.used()

    def test_distributed_cluster_initialization(self):
        """Test DistributedCluster initialization."""
        with patch.dict(
            os.environ,
            {"WORLD_SIZE": "8", "RANK": "3", "LOCAL_RANK": "1", "MASTER_ADDR": "192.168.1.1", "MASTER_PORT": "29500"},
        ):
            cluster = DistributedCluster()

            assert cluster.world_size == 8
            assert cluster.global_rank == 3
            assert cluster.local_rank == 1
            assert cluster.master_addr == "192.168.1.1"
            assert cluster.master_port == 29500


class TestMappingCluster:
    """Tests for MappingCluster (custom mapping)."""

    def test_mapping_cluster_with_dict(self):
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
            cluster = MappingCluster(mapping=mapping)

            assert cluster.world_size == 8
            assert cluster.global_rank == 3
            assert cluster.local_rank == 1
            assert cluster.master_addr == "192.168.1.1"
            assert cluster.master_port == 29500

    def test_mapping_cluster_with_env_mapping(self):
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
            cluster = MappingCluster(mapping=mapping)

            assert cluster.world_size == 4
            assert cluster.global_rank == 2
            assert cluster.local_rank == 0
            assert cluster.init_method == "env://"

    def test_mapping_cluster_defaults(self):
        """Test MappingCluster with missing environment variables."""
        mapping = EnvMapping(
            local_rank="MY_LOCAL_RANK",
            global_rank="MY_GLOBAL_RANK",
            world_size="MY_WORLD_SIZE",
            master_addr="MY_MASTER_ADDR",
            master_port="MY_MASTER_PORT",
        )

        with patch.dict(os.environ, {}, clear=True):
            cluster = MappingCluster(mapping=mapping)

            # Should use defaults
            assert cluster.world_size == 1
            assert cluster.global_rank == 0
            assert cluster.local_rank == 0
            assert cluster.master_addr == ""
            assert cluster.master_port == 0

    def test_mapping_cluster_not_used(self):
        """Test that MappingCluster.used() returns False."""
        assert not MappingCluster.used()

    def test_mapping_cluster_list(self):
        """Test MappingCluster with list mapping, such that the first found env var is used."""
        mapping = EnvMapping(
            local_rank=["MY_LOCAL_RANK_A", "MY_LOCAL_RANK_B"],
            global_rank=["MY_GLOBAL_RANK_A", "MY_GLOBAL_RANK_B"],
            world_size=["MY_WORLD_SIZE_A", "MY_WORLD_SIZE_B"],
            master_addr=["MY_MASTER_ADDR_A", "MY_MASTER_ADDR_B"],
            master_port=["MY_MASTER_PORT_A", "MY_MASTER_PORT_B"],
            init_method="tcp://{master_addr}:{master_port}",
        )

        with patch.dict(
            os.environ,
            {
                "MY_WORLD_SIZE_B": "16",
                "MY_GLOBAL_RANK_A": "5",
                "MY_LOCAL_RANK_B": "2",
                "MY_MASTER_ADDR_A": "192.168.1.1",
                "MY_MASTER_PORT_B": "40000",
            },
        ):
            cluster = MappingCluster(mapping=mapping)

            assert cluster.world_size == 16
            assert cluster.global_rank == 5
            assert cluster.local_rank == 2
            assert cluster.master_addr == "192.168.1.1"
            assert cluster.master_port == 40000


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

    def test_create_cluster_with_config(self):
        """Test create_cluster with explicit config."""
        config = {"manual": {"world_size": 4}}

        # Ensure the environment marker is not set
        with patch.dict(os.environ, {}, clear=True):
            # create_cluster with manual config returns a ManualSpawner
            cluster = create_cluster(config)

            assert isinstance(cluster, ManualSpawner)
            assert cluster._world_size == 4

    def test_create_cluster_auto_detection(self):
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
            cluster = create_cluster({})

            assert isinstance(cluster, SlurmCluster)
            assert cluster.world_size == 4

    def test_create_cluster_mpi_detection(self):
        """Test create_cluster with MPI detection."""
        with patch.dict(
            os.environ,
            {
                "OMPI_COMM_WORLD_SIZE": "8",
                "OMPI_COMM_WORLD_RANK": "0",
                "OMPI_COMM_WORLD_LOCAL_RANK": "0",
                "MASTER_ADDR": "192.168.1.1",
                "MASTER_PORT": "29500",
            },
        ):
            cluster = create_cluster({})

            assert isinstance(cluster, MPICluster)
            assert cluster.world_size == 8

    def test_create_cluster_slurm_with_pmi_set(self):
        """Test create_cluster with Slurm detection when PMI variables are set."""
        with patch.dict(
            os.environ,
            {
                "SLURM_NTASKS": "4",
                "SLURM_PROCID": "0",
                "SLURM_LOCALID": "0",
                "SLURM_NODELIST": "node001",
                "SLURM_JOBID": "12345",
                "SLURM_JOB_NAME": "test_job",
                "PMI_SIZE": "4",
                "PMI_RANK": "0",
                "MASTER_ADDR": "192.168.1.1",
                "MASTER_PORT": "29500",
            },
        ):
            cluster = create_cluster({})

            assert isinstance(cluster, SlurmCluster)
            assert cluster.world_size == 4

    def test_create_cluster_no_suitable_cluster(self):
        """Test create_cluster when no suitable cluster found."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="No suitable cluster found"):
                create_cluster({})


class TestClusterBase:
    """Tests for base Cluster class functionality."""

    def test_cluster_init_method(self):
        """Test cluster init_method property."""
        client = ManualClient(
            world_size=4,
            local_rank=0,
            global_rank=0,
            master_addr="localhost",
            master_port=12345,
        )

        init_method = client.init_method
        assert init_method.startswith("tcp://")
        assert "localhost" in init_method
        assert str(client.master_port) in init_method

    def test_cluster_backend_cuda(self, mock_context):
        """Test cluster backend selection for CUDA."""
        mock_context.device.type = "cuda"
        with patch("anemoi.inference.lazy.torch.cuda.is_available", return_value=True):
            client = ManualClient(
                world_size=4,
                local_rank=0,
                global_rank=0,
                master_addr="localhost",
                master_port=12345,
            )
            assert client.backend == "nccl"

    def test_cluster_backend_cpu(self):
        """Test cluster backend selection for CPU."""
        with patch("anemoi.inference.lazy.torch.cuda.is_available", return_value=False):
            client = ManualClient(
                world_size=4,
                local_rank=0,
                global_rank=0,
                master_addr="localhost",
                master_port=12345,
            )
            assert client.backend == "gloo"

    def test_cluster_is_master(self):
        """Test cluster is_master property."""
        client0 = ManualClient(
            world_size=4,
            local_rank=0,
            global_rank=0,
            master_addr="localhost",
            master_port=12345,
        )
        assert client0.is_master

        client1 = ManualClient(
            world_size=4,
            local_rank=1,
            global_rank=1,
            master_addr="localhost",
            master_port=12345,
        )
        assert not client1.is_master
