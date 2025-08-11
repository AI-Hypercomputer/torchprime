"""
Parse a profile to determine the median duration of a training step.
"""

import glob
import logging
import os
import statistics
import sys
from datetime import datetime
from urllib.parse import urlparse

from torchprime.metrics.xplane_pb2 import XSpace  # type: ignore

logger = logging.getLogger(__name__)


def get_latest_profile_path(profile_dir: str) -> str:
  """Finds the most recently updated .xplane.pb file in a directory or GCS bucket.

  Args:
      profile_dir: Local directory path (e.g., '/path/to/profiles/') or GCS path (e.g., 'gs://my-data-bucket/profiles/').

  Returns:
      Path to the newest .xplane.pb file (local path for local files, gs:// path for GCS).

  Raises:
      ValueError: If no .xplane.pb files are found.
  """
  if profile_dir.startswith("gs://"):
    from google.cloud import storage

    # Parse GCS path
    parsed = urlparse(profile_dir, scheme="gs")
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/") + "/" if parsed.path.strip("/") else ""

    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # List objects matching the glob pattern
    blobs = bucket.list_blobs(match_glob=f"{prefix}**/*.xplane.pb")

    # Find the blob with the latest updated timestamp
    newest_blob = None
    latest_time = datetime.min
    for blob in blobs:
      if blob.updated > latest_time:
        newest_blob = blob
        latest_time = blob.updated

    if not newest_blob:
      raise ValueError(f"No .xplane.pb files found in {profile_dir}")

    # Return gs:// path
    return f"gs://{bucket_name}/{newest_blob.name}"

  else:
    # Local filesystem path
    profile_dir = os.path.abspath(profile_dir)
    profiles = [
      (f, os.path.getctime(f))
      for f in glob.glob(f"{profile_dir}/**/*.xplane.pb", recursive=True)
    ]
    if not profiles:
      raise ValueError(f"No .xplane.pb files found in {profile_dir}")
    newest_profile, _time = max(profiles, key=lambda v: v[1])
    return newest_profile


def analyze_step_duration(file_path: str) -> float:
  """Analyzes the step duration from an .xplane.pb file.

  Args:
      file_path: Path to the .xplane.pb file (local path or gs:// path).

  Returns:
      Float value representing the step duration.
  """
  xspace = XSpace()

  if file_path.startswith("gs://"):
    from google.cloud import storage

    # Parse GCS path
    parsed = urlparse(file_path, scheme="gs")
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")

    # Read file content from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    logger.info("Downloading %s", file_path)
    file_content = blob.download_as_bytes()
  else:
    # Read local file
    logger.info("Loading %s", file_path)
    with open(file_path, "rb") as f:
      file_content = f.read()

  # Parse the xplane proto
  xspace.ParseFromString(file_content)
  return analyze_step_duration_from_pb(xspace)


def step_duration_from_latest_profile(profile_dir: str) -> float:
  """Finds the most recently updated .xplane.pb file in a directory or GCS bucket and analyzes its step duration.

  Args:
      profile_dir: Local directory path (e.g., '/path/to/profiles/') or GCS path (e.g., 'gs://my-data-bucket/profiles/').

  Returns:
      Float value from analyze_step_duration for the newest profile.
  """
  file_path = get_latest_profile_path(profile_dir)
  logging.info("Found newest profile: %s", file_path)
  return analyze_step_duration(file_path)


def analyze_step_duration_from_pb(xspace: XSpace) -> float:
  offsets = []
  unique_names = set()

  for plane in xspace.planes:
    # Only consider /device:TPU:0
    if plane.name != "/device:TPU:0":
      continue
    print(f"Plane ID: {plane.id}, Name: {plane.name}", file=sys.stderr)

    for line in plane.lines:
      # Only consider XLA Modules line
      if line.name != "XLA Modules":
        continue
      print(f"  Line ID: {line.id}, Name: {line.name}", file=sys.stderr)

      # Collect offsets and event names
      for event in line.events:
        name = plane.event_metadata[event.metadata_id].name
        offset_ps = event.offset_ps
        unique_names.add(name)
        offsets.append(offset_ps)
        print(
          f"    Event Metadata Name: {name}, "
          f"ID: {event.metadata_id}, Offset: {offset_ps / 1e12:.3f} s, "
          f"Duration: {event.duration_ps / 1e12:.3f} s",
          file=sys.stderr,
        )

  # Make sure we have events at all
  if not offsets:
    raise ValueError("No events found in the given XSpace data.")

  # Confirm we have exactly one unique event name
  if len(unique_names) > 1:
    raise ValueError(f"Ambiguous event names found in XSpace: {unique_names}")

  inferred_event_name = max(unique_names)

  # Sort offsets to compute consecutive differences
  offsets.sort()

  if len(offsets) < 2:
    raise ValueError("Not enough events to compute step durations.")

  # Compute durations based on consecutive offset differences
  durations = []
  for i in range(len(offsets) - 1):
    # Convert picoseconds to seconds
    durations.append((offsets[i + 1] - offsets[i]) / 1e12)

  # If we have no intervals, we can't compute durations
  event_count = len(durations)
  if event_count == 0:
    raise ValueError("Not enough events to compute step durations.")

  print(
    f"Got {event_count} intervals for event '{inferred_event_name}'", file=sys.stderr
  )

  # If fewer than 3 intervals, compute a simple average
  if event_count < 3:
    print(
      "[Warning] Not enough events found to drop outliers.",
      file=sys.stderr,
    )
    return sum(durations) / len(durations)

  # Otherwise, use the median
  average_duration = statistics.median(durations)
  return average_duration


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <path_to_proto_file>")
    sys.exit(1)
  proto_file_path = sys.argv[1]
  try:
    median_duration = analyze_step_duration(proto_file_path)
    print(f"Median step duration: {median_duration:.4f}")
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
