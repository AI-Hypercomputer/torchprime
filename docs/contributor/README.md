## Contributing to torchprime

When developing, use `pip install -e '.[dev]'` to install dev dependencies such
as linter and formatter.

### How to run tests

```sh
pytest
```

### How to run some of the tests, and re-run them whenever you change a file

```sh
tp -i test ... # replace with path to tests/directories
```

### How to format

```sh
ruff format
```

### How to lint

```sh
ruff check [--fix]
```

You can install a Ruff VSCode plugin to check errors and format files from the
editor.
