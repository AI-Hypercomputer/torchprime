# Troubleshooting distributed setup

`torchprime` is designed for scaled, distributed training. 
Once properly configured, the `tp run` command-line function 
will coordinate several infrastructure tools to move your
training code onto a cluster, train, and log results. 

However, the trade-off is the investment of time
getting `tp` configured to properly coordinate with your cluster. 

## tp doctor

To validate that `tp` is configured correctly, run

```sh
tp doctor
```

This runs through a series of checks, highlighting additional
configurations you may need to run.

## End-to-end test

A simple first script to run using `tp run` is the `system_check.py` script. 
This script will log data about the host, and attempt to run minimal
calculation with PyTorch using both the CPU and XLA backends. 

```sh
tp run torchprime/tools/system_check.py --name first-run
```

The logs are sent to both the cluster's stdout, which you can retrieve 
in Google Cloud Logs Explorer
via the link provided in the stdout of your local machine where you
ran `tp run`, as well as the bucket you configured via the 
`--artifact-dir` flag of `tp use`. Specifically, you'll find the logs in

`<artifact-dir><run-name><outputs><<slice>-<host>>log.log`.

An example output: 

```log
06/02/2025 21:09:12 - INFO - __main__ - Attached logger to file: /tmp/gcs-mount/first-run/outputs/0-0/log.log
06/02/2025 21:09:12 - INFO - __main__ - ======================================================================
06/02/2025 21:09:12 - INFO - __main__ -                        PyTorch Environment Information                
06/02/2025 21:09:12 - INFO - __main__ - ======================================================================
06/02/2025 21:09:12 - INFO - __main__ - --- Command Line Arguments ---
06/02/2025 21:09:12 - INFO - __main__ - Argument 0: torchprime/tools/system_check.py
06/02/2025 21:09:12 - INFO - __main__ - Argument 1: profile_dir=/tmp/gcs-mount/first-run/profile/0-0
06/02/2025 21:09:12 - INFO - __main__ - Argument 2: output_dir=/tmp/gcs-mount/first-run/outputs/0-0
06/02/2025 21:09:12 - INFO - __main__ - Total arguments: 3
06/02/2025 21:09:12 - INFO - __main__ - ----------------------------------------------------------------------
06/02/2025 21:09:12 - INFO - __main__ - --- Basic System Information ---
06/02/2025 21:09:12 - INFO - __main__ - Timestamp: 2025-06-02T21:09:12.949352
06/02/2025 21:09:12 - INFO - __main__ - Hostname: gke-tpu-cad093a4-mk21
06/02/2025 21:09:13 - INFO - __main__ - Platform: Linux-6.6.72+-x86_64-with-glibc2.31
06/02/2025 21:09:13 - INFO - __main__ - Machine: x86_64
06/02/2025 21:09:13 - INFO - __main__ - Processor: 
06/02/2025 21:09:13 - INFO - __main__ - Architecture: 64bit
06/02/2025 21:09:13 - INFO - __main__ - System: Linux 6.6.72+
06/02/2025 21:09:13 - INFO - __main__ - Version: #1 SMP PREEMPT_DYNAMIC Mon Feb 17 11:14:31 UTC 2025
06/02/2025 21:09:13 - INFO - __main__ - OS: Linux 6.6.72+ (#1 SMP PREEMPT_DYNAMIC Mon Feb 17 11:14:31 UTC 2025)
06/02/2025 21:09:13 - INFO - __main__ - Python Version: 3.10.17 (main, Apr 29 2025, 00:24:14) [GCC 10.2.1 20210110]
06/02/2025 21:09:13 - INFO - __main__ - Python Executable: /usr/local/bin/python
06/02/2025 21:09:13 - INFO - __main__ - ----------------------------------------------------------------------
06/02/2025 21:09:13 - INFO - __main__ - --- All Environment Variables ---
06/02/2025 21:09:13 - INFO - __main__ - ALT=false
06/02/2025 21:09:13 - INFO - __main__ - CHIPS_PER_HOST_BOUNDS=2,2,1
06/02/2025 21:09:13 - INFO - __main__ - GPG_KEY=ABC123
06/02/2025 21:09:13 - INFO - __main__ - HOME=/root
06/02/2025 21:09:13 - INFO - __main__ - HOSTNAME=gke-tpu-abc-123
06/02/2025 21:09:13 - INFO - __main__ - HOST_BOUNDS=2,2,1
06/02/2025 21:09:13 - INFO - __main__ - JOB_COMPLETION_INDEX=0
06/02/2025 21:09:13 - INFO - __main__ - KUBERNETES_PORT=tcp://1.1.1.1:443
06/02/2025 21:09:13 - INFO - __main__ - KUBERNETES_PORT_443_TCP=tcp://1.1.1.1:443
06/02/2025 21:09:13 - INFO - __main__ - KUBERNETES_PORT_443_TCP_ADDR=1.1.1.1
06/02/2025 21:09:13 - INFO - __main__ - KUBERNETES_PORT_443_TCP_PORT=443
06/02/2025 21:09:13 - INFO - __main__ - KUBERNETES_PORT_443_TCP_PROTO=tcp
06/02/2025 21:09:13 - INFO - __main__ - KUBERNETES_SERVICE_HOST=1.1.1.1
06/02/2025 21:09:13 - INFO - __main__ - KUBERNETES_SERVICE_PORT=443
06/02/2025 21:09:13 - INFO - __main__ - KUBERNETES_SERVICE_PORT_HTTPS=443
06/02/2025 21:09:13 - INFO - __main__ - LANG=C.UTF-8
06/02/2025 21:09:13 - INFO - __main__ - LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --megascale_grpc_enable_xor_tracer=false
06/02/2025 21:09:13 - INFO - __main__ - MEGASCALE_COORDINATOR_ADDRESS=first-run-slice-job-0-0.first-run
06/02/2025 21:09:13 - INFO - __main__ - MEGASCALE_NUM_SLICES=1
06/02/2025 21:09:13 - INFO - __main__ - MEGASCALE_SLICE_ID=0
06/02/2025 21:09:13 - INFO - __main__ - PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
06/02/2025 21:09:13 - INFO - __main__ - PWD=/workspaces/torchprime
06/02/2025 21:09:13 - INFO - __main__ - PYTHON_SHA256=abc123
06/02/2025 21:09:13 - INFO - __main__ - PYTHON_VERSION=3.10.17
06/02/2025 21:09:13 - INFO - __main__ - SHLVL=1
06/02/2025 21:09:13 - INFO - __main__ - TF_CPP_MIN_LOG_LEVEL=0
06/02/2025 21:09:13 - INFO - __main__ - TORCHPRIME_ARTIFACT_DIR=gs://first-run
06/02/2025 21:09:13 - INFO - __main__ - TORCHPRIME_JOBSET_NAME=first-run
06/02/2025 21:09:13 - INFO - __main__ - TPU_ACCELERATOR_TYPE=v6e-16
06/02/2025 21:09:13 - INFO - __main__ - TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
06/02/2025 21:09:13 - INFO - __main__ - TPU_HOST_BOUNDS=2,2,1
06/02/2025 21:09:13 - INFO - __main__ - TPU_MIN_LOG_LEVEL=0
06/02/2025 21:09:13 - INFO - __main__ - TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434
06/02/2025 21:09:13 - INFO - __main__ - TPU_SKIP_MDS_QUERY=true
06/02/2025 21:09:13 - INFO - __main__ - TPU_STDERR_LOG_LEVEL=0
06/02/2025 21:09:13 - INFO - __main__ - TPU_TOPOLOGY=4x4
06/02/2025 21:09:13 - INFO - __main__ - TPU_TOPOLOGY_ALT=false
06/02/2025 21:09:13 - INFO - __main__ - TPU_TOPOLOGY_WRAP=false,false,false
06/02/2025 21:09:13 - INFO - __main__ - TPU_VMODULE=real_program_continuator=1
06/02/2025 21:09:13 - INFO - __main__ - TPU_WORKER_HOSTNAMES=first-run-slice-job-0-0.first-run,first-run-slice-job-0-1.first-run,first-run-slice-job-0-2.first-run,first-run-slice-job-0-3.first-run
06/02/2025 21:09:13 - INFO - __main__ - TPU_WORKER_ID=0
06/02/2025 21:09:13 - INFO - __main__ - VBAR_CONTROL_SERVICE_URL=1.1.1.1:8353
06/02/2025 21:09:13 - INFO - __main__ - WRAP=false,false,false
06/02/2025 21:09:13 - INFO - __main__ - XLA_FLAGS= --xla_dump_to=/tmp/gcs-mount/first-run/xla_dumps/0-0/ --xla_dump_hlo_as_proto --xla_dump_hlo_as_text
06/02/2025 21:09:13 - INFO - __main__ - _=/usr/local/bin/python
06/02/2025 21:09:13 - INFO - __main__ - Logged 45 environment variable(s).
06/02/2025 21:09:13 - INFO - __main__ - ----------------------------------------------------------------------
06/02/2025 21:09:13 - INFO - __main__ - --- PyTorch Information ---
06/02/2025 21:09:14 - INFO - __main__ - torch imported successfully.
06/02/2025 21:09:14 - INFO - __main__ - torch version: 2.8.0
06/02/2025 21:09:14 - INFO - __main__ - torch CPU tensor addition (a+b): tensor([1., 2., 3.]) + tensor([4., 5., 6.]) = tensor([5., 7., 9.])
06/02/2025 21:09:14 - INFO - __main__ - torch.cuda.is_available(): False
06/02/2025 21:09:14 - INFO - __main__ - torch.backends.mps.is_available(): False (torch.backends.mps.is_built(): False)
06/02/2025 21:09:14 - INFO - __main__ - --- PyTorch/XLA Information ---
06/02/2025 21:09:16 - INFO - __main__ - torch_xla imported successfully.
06/02/2025 21:09:16 - INFO - __main__ - torch_xla version: 2.8.0+git64a676c
06/02/2025 21:09:48 - INFO - __main__ - torch_xla tensor addition on xla (a+b): tensor([10., 20.], device='xla:0') + tensor([30., 40.], device='xla:0') = tensor([40., 60.], device='xla:0')
06/02/2025 21:09:48 - INFO - __main__ - ----------------------------------------------------------------------
06/02/2025 21:09:48 - INFO - __main__ - ======================================================================
06/02/2025 21:09:48 - INFO - __main__ -                   End of PyTorch Environment Logging                  
06/02/2025 21:09:48 - INFO - __main__ - ======================================================================
```

## FAQ

### `tp doctor` indicates setup is correct, but I get an authentication error when I run `tp run`


If you see an error like this despite a successful run of
`tp doctor`, 

```
unauthorized: authentication failed
Error running command: Command '['sudo', 'docker', 'push', ... 
```

this may be because `tp` is using sudo to push docker images, but your
gcloud is not authenticated as root. Either setup 
[non-root docker](https://docs.docker.com/engine/install/linux-postinstall/)
or authenticate gcloud as root.

```sh
sudo gcloud auth login
```