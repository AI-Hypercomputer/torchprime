import csv
import pprint

import metrics_log


def test_metrics_logger(tmpdir):
    # Arrange
    tmppath = tmpdir.join("unit_test_metrics_logger.csv")
    run_name1 = "test_run1"
    run_name2 = "test_run2"
    hyperparameters1 = {"lr": 0.011, "batch_size": 11}
    hyperparameters2 = {"lr": 0.022, "batch_size": 22}

    metrics_1_1 = {"accuracy": 1.11, "perplexity": 1.111}
    metrics_1_2 = {"accuracy": 1.22, "perplexity": 1.222}
    metrics_2_1 = {"accuracy": 2.11, "perplexity": 2.111}
    metrics_2_2 = {"accuracy": 2.22, "perplexity": 2.222}
    metrics_2_3 = {"accuracy": 2.33, "perplexity": 2.333}

    # Act
    logger1 = metrics_log.MetricsLogger(run_name1, hyperparameters1, path=tmppath)
    logger2 = metrics_log.MetricsLogger(run_name2, hyperparameters2, path=tmppath)

    logger1.log(1, metrics_1_1)
    logger2.log(1, metrics_2_1)
    logger2.log(2, metrics_2_2)
    logger2.log(3, metrics_2_3)
    logger1.log(2, metrics_1_2)

    # Assert
    with open(tmppath, "r") as f:
        lines = list(csv.reader(f))
    assert len(lines) == 6  # Header + 5 data rows
    assert (
        ",".join(lines[0])
        == "run_name,datetime,hyperparameters,epoch,accuracy,perplexity"
    )

    assert lines[1][0] == run_name1
    assert lines[2][0] == run_name2
    assert lines[3][0] == run_name2
    assert lines[4][0] == run_name2
    assert lines[5][0] == run_name1

    assert lines[1][2] == pprint.pformat(hyperparameters1)
    assert lines[2][2] == pprint.pformat(hyperparameters2)
    assert lines[3][2] == pprint.pformat(hyperparameters2)
    assert lines[4][2] == pprint.pformat(hyperparameters2)
    assert lines[5][2] == pprint.pformat(hyperparameters1)

    assert lines[1][3] == "1"
    assert lines[2][3] == "1"
    assert lines[3][3] == "2"
    assert lines[4][3] == "3"
    assert lines[5][3] == "2"

    assert lines[1][4] == str(metrics_1_1["accuracy"])
    assert lines[2][4] == str(metrics_2_1["accuracy"])
    assert lines[3][4] == str(metrics_2_2["accuracy"])
    assert lines[4][4] == str(metrics_2_3["accuracy"])
    assert lines[5][4] == str(metrics_1_2["accuracy"])

    assert lines[1][5] == str(metrics_1_1["perplexity"])
    assert lines[2][5] == str(metrics_2_1["perplexity"])
    assert lines[3][5] == str(metrics_2_2["perplexity"])
    assert lines[4][5] == str(metrics_2_3["perplexity"])
    assert lines[5][5] == str(metrics_1_2["perplexity"])
