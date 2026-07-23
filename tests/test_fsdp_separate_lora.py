import subprocess

import pytest

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_fsdp_separate_lora_updates_only_active_adapter(tmp_path):
    """Two-GPU smoke test for post-FSDP adapter switching and isolation."""

    if current_platform.device_count() < 2:
        pytest.skip("Distributed separate-LoRA test requires 2 GPUs")
    output = tmp_path / "fsdp_separate_lora.out"
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                "--nproc_per_node=2",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_fsdp_separate_lora.py",
                "--backend=fsdp:d2t1c1",
                f"--output={output}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(f"Test failed: {error.stderr}\n{error.stdout}")
    assert output.read_text().strip() == "Passed"
