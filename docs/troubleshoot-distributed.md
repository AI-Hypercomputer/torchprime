# Troubleshooting Distributed Setup

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

A simple first script to run using `tp run` is the log_and_exit script. 
This script will log data about the host, and attempt to run a simple
calculations with PyTorch using both the CPU and XLA backends. 

The logs are sent to both the cluster's stdout, which you can retrieve 
in Google Cloud Logs Explorer
via the link provided in the stdout of your local machine where you
ran `tp run`, as well as the bucket you configured via the 
`--artifact-dir` flag of `tp use`. Specifically, you'll find the logs in

`<artifact-dir><run-name><outputs><<slice>-<host>>log.log`.

## FAQ

### `tp doctor` indicates setup is correct, but I get an authentication error when I run `tp run`


If you see an error like this despite a successful run of
`tp doctor`, 

```
unauthorized: authentication failed
Error running command: Command '['sudo', 'docker', 'push', ... 
```

you may need to authenticate your gcloud as root.

```sh
sudo gcloud auth login
```