import os
import subprocess
import sys
from datetime import datetime

# Some passes in XLA can only dump to regular folders. Hence,
# mount the GCS bucket at gs://hanq-llama at `/tmp/gcs-mount` using gcsfuse.
gcs_mount = "/tmp/gcs-mount"
os.makedirs(gcs_mount, exist_ok=True)
subprocess.run(["gcsfuse", "hanq-llama", gcs_mount], check=True)

# These are set by GKE automatically.
worker_id = os.getenv("TPU_WORKER_ID", "0")
slice_id = os.getenv("MEGASCALE_SLICE_ID", "0")

# Configure XLA graph dump path before doing anything else.
date_string = datetime.now().strftime("%Y%m%d-%H%M")
jobset_name = os.getenv("JOBSET_NAME", date_string)
xla_dump_path = (
    f"{gcs_mount}/llama3-{slice_id}-{worker_id}/xla_dumps/{jobset_name}/")
os.environ["XLA_FLAGS"] = (
    os.getenv("XLA_FLAGS", "") + f" --xla_dump_to={xla_dump_path}")
print(f"Dumping XLA compiler outputs to {xla_dump_path}")

# Determine the profile dir
profile_dir = f"{gcs_mount}/llama3-{slice_id}-{worker_id}/"

# Exec into the training script.
args = [sys.executable] + sys.argv[1:] + ["--profile_dir", profile_dir]
env = os.environ.copy()
os.execve(sys.executable, args, env)
