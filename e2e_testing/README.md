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

[e2e-test]: /.github/workflows/e2e_test.yml
[e2e-check]: /.github/workflows/reusable_e2e_check.yml
