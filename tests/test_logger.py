import logging

from utils.config import load_config
from utils.logger import (
    config_hash,
    get_logger,
    log_run_header,
    setup_logging,
)


def test_setup_logging_returns_logger(tmp_path):
    logger = setup_logging("INFO", log_file=tmp_path / "run.log")
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO


def test_logger_writes_to_console_and_file(tmp_path):
    log_file = tmp_path / "run.log"
    logger = setup_logging("INFO", log_file=log_file)
    logger.info("hello marker")

    content = log_file.read_text(encoding="utf-8")
    assert "hello marker" in content


def test_setup_logging_is_idempotent(tmp_path):
    setup_logging("INFO", log_file=tmp_path / "a.log")
    handler_count = len(logging.getLogger("cvproject").handlers)

    setup_logging("INFO", log_file=tmp_path / "a.log")
    assert len(logging.getLogger("cvproject").handlers) == handler_count


def test_get_logger_has_configured_propagation():
    logger = get_logger("tests.child")
    assert logger.name == "cvproject.tests.child"
    assert len(logging.getLogger("cvproject").handlers) >= 1


def test_config_hash_deterministic():
    cfg = load_config("pipeline")
    assert config_hash(cfg) == config_hash(cfg)


def test_config_hash_differs_on_change():
    a = load_config("unet")
    b = load_config("unet", overrides={"train": {"epochs": 61}})
    assert config_hash(a) != config_hash(b)


def test_log_run_header_emits_fingerprint(tmp_path):
    logger = setup_logging("INFO", log_file=tmp_path / "hdr.log")
    log_run_header(logger, load_config("pipeline"), title="TESTRUN")
    content = tmp_path.joinpath("hdr.log").read_text(encoding="utf-8")
    assert "TESTRUN" in content
    assert "config_hash=" in content


def test_setup_logging_without_file():
    logger = setup_logging()
    assert logger is not None