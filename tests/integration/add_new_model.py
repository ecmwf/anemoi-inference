import argparse
import json
import re
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import yaml
from anemoi.utils.checkpoints import load_metadata
from anemoi.utils.remote import transfer
from anemoi.utils.remote.s3 import _list_objects
from anemoi.utils.sanitise import sanitise

from anemoi.inference.testing import save_fake_checkpoint

warnings.filterwarnings(
    "ignore",
    message=".*DotDict.*",  # ignore DotDict immutable warning triggered by `transfer`
    category=UserWarning,
    module="anemoi.utils.config",
)

INTEGRATION_ROOT = Path(__file__).resolve().parent
S3_ROOT = "s3://ml-tests/test-data/samples/anemoi-integration-tests/inference"

parser = argparse.ArgumentParser(
    description="Add a new model for integration tests. Running this script will create a new model directory in tests/integration with metadata and config files, and upload necessary files to S3."
)
parser.add_argument(
    "model",
    type=str,
    help="Name of the model. Can only contain alphanumeric, underscores, hyphens, or dots.",
)
parser.add_argument(
    "checkpoint",
    type=Path,
    help="Path to the inference checkpoint file.",
)
parser.add_argument(
    "--files",
    "-f",
    type=Path,
    nargs="*",
    default=[],
    help="Additional files to upload to the model directory on S3.",
)
parser.add_argument(
    "--overwrite",
    "-o",
    action="store_true",
    help="Overwrite existing files.",
)
parser.add_argument(
    "--save-fake-checkpoint",
    action="store_true",
    help="Save a fake checkpoint file locally alongside the real checkpoint for testing purposes.",
)
args = parser.parse_args()

model_path = INTEGRATION_ROOT / args.model
metadata_path = model_path / "metadata.json"
config_path = model_path / "config.yaml"

if not re.match(r"^[\w\-\.]+$", args.model):
    print("Model name must be a valid path name (alphanumeric, underscores, hyphens, or dots only).")
    sys.exit(1)

if not args.checkpoint.exists():
    print(f"Checkpoint file {args.checkpoint} does not exist.")
    sys.exit(1)

for file in args.files:
    if not file.exists():
        print(f"{file} does not exist.")
        sys.exit(1)

if model_path.exists() and not args.overwrite:
    print(f"Model directory `{args.model}` already exists in `{INTEGRATION_ROOT}`. Use --overwrite to replace it.")
    sys.exit(1)

model_path.mkdir(parents=True, exist_ok=True)

print(f"🚀 Adding model {args.model}...")
metadata, supporting_arrays = load_metadata(args.checkpoint, supporting_arrays=True)
metadata = sanitise(metadata, level=2)

# delete unused data
del metadata["tracker"]
del metadata["training"]
del metadata["model"]
metadata["config"] = {
    "data": metadata["config"]["data"],
    "training": metadata["config"]["training"],
}

if args.save_fake_checkpoint:
    fake_checkpoint_path = Path(args.checkpoint).parent / f"{args.model}-fake.ckpt"
    print(f"💾 Saving fake checkpoint to {fake_checkpoint_path}")
    save_fake_checkpoint(metadata, fake_checkpoint_path, supporting_arrays=supporting_arrays)

# save metadata and example config file in the local repo
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

with open(config_path, "w") as f:
    yaml.dump(
        [
            {
                "name": "example-integration-test",
                "input": "input.grib",
                "output": "output.grib",
                "checks": [],
                "inference_config": {
                    "checkpoint": "${checkpoint:}",
                    "input": {"grib": "${input:}"},
                    "output": {"grib": "${output:}"},
                },
            }
        ],
        f,
        sort_keys=False,
    )

# upload files to S3
for file in args.files:
    if file.is_dir():
        for subfile in file.iterdir():
            s3_path = f"{S3_ROOT}/{args.model}/{file}/{subfile.name}"
            transfer(str(subfile), s3_path, overwrite=args.overwrite, resume=not args.overwrite)
    else:
        s3_path = f"{S3_ROOT}/{args.model}/{file.name}"
        transfer(str(file), s3_path, overwrite=args.overwrite, resume=not args.overwrite)

with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir_path = Path(temp_dir)
    for name, array in supporting_arrays.items():
        array_path = temp_dir_path / f"{name}.npy"
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(array_path, array)
        s3_path = f"{S3_ROOT}/{args.model}/supporting-arrays/{name}.npy"
        transfer(
            str(array_path),
            s3_path,
            overwrite=args.overwrite,
            resume=not args.overwrite,
        )

print("Done. Summary:")
print(f"💾 Files to be comitted, created in {model_path}:")
for item in model_path.iterdir():
    print(f" - {item.name}")

print(f"☁️ Files in {S3_ROOT}/{args.model}:")
for item in _list_objects(f"{S3_ROOT}/{args.model}"):
    print(f" - {item['path'].split(f'{args.model}/')[-1]}")

print(f"✅ Model {args.model} added successfully.")
