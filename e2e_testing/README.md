# E2E testing

These scripts are used during the [E2E test][e2e-test] GitHub action to run some
models and validate the results.

## E2E test design

The workflows in [e2e_test.yml][e2e-test] does a few things:

- Set up `gcloud` credentials from a Service Account key managed in repo secrets.
- Install `torchprime`.
- Test `tp use` and point it to an XPK cluster hosted internally.
- Test `tp run` on a few models.

After kicking off the training of some models, it starts a parallel job for each
model, and runs a few checks. This is implemented in
[reusable_e2e_check.yml][e2e-check]:

- Stream the logs.
- Check workload exit code.
- Check for specific log strings that indicate training success.
- Check that there is a profile `.pb` file.
- Check that the step time is within a reasonable range.

## What to do when step time is out of range

As of https://github.com/AI-Hypercomputer/torchprime/pull/208, we'll start
checking that the step time of model training E2E tests falls within
experimentally derived bounds.

### If the step time falls below the lower bound

If this is the result of an optimization/updated deps, you may re-center the
bounds to the latest step time, keeping the confidence intervals unchanged.

If there were no code changes and the step time is still below the lower bound,
consider growing the confidence interval.

### If the step time goes above the upper bound

If this is the result of a code change, you may have introduced a regression.
Investigate the root cause and avoid the slow down before landing the PR.

If there were no code changes and the step time is still above the upper bound,
we'll need to discuss with the hardware teams because it may be the result of
hardware changes.

## v6e XPK cluster

E2E tests are launched onto an XPK cluster named `tpu-v6e-ci`.

(Googlers only) To heal or re-create the cluster, use the following:

```sh
xpk cluster create \
    --tpu-type v6e-4 \
    --cluster tpu-v6e-ci \
    --num-slices 64 \
    --on-demand \
    --zone us-central2-b \
    --project tpu-pytorch \
    --default-pool-cpu-machine-type=n2-standard-32
```

[e2e-test]: /.github/workflows/e2e_test.yml
[e2e-check]: /.github/workflows/reusable_e2e_check.yml
