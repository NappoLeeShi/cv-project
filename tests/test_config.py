import pytest

from utils.config import (
    CONFIG_DIR,
    ROOT_DIR,
    ConfigError,
    clear_config_cache,
    config_path,
    get,
    load_config,
    resolve_path,
)


@pytest.fixture(autouse=True)
def _reset_config_cache():
    clear_config_cache()
    yield
    clear_config_cache()


def test_root_dir_points_to_project_root():
    assert (ROOT_DIR / "main.py").is_file()
    assert (ROOT_DIR / "configs").is_dir()
    assert (ROOT_DIR / "utils" / "config.py").is_file()
    assert ROOT_DIR.name == "cv-project"


def test_config_dir_exists():
    assert CONFIG_DIR.is_dir()


def test_all_config_files_exist():
    for name in ("unet.yaml", "midas.yaml", "pipeline.yaml"):
        assert config_path(name).is_file(), f"missing {name}"


def test_config_path_normalizes_extension():
    assert config_path("unet") == config_path("unet.yaml")


def test_load_unet_config_values():
    cfg = load_config("unet")
    assert cfg["model"]["num_classes"] == 19
    assert cfg["model"]["base_channels"] == 64
    assert cfg["data"]["image_size"] == [512, 1024]
    assert cfg["env"]["seed"] == 42
    assert cfg["env"]["device"] == "auto"
    assert cfg["train"]["epochs"] == 60
    assert cfg["train"]["optimizer"] == "adam"


def test_load_midas_config_values():
    cfg = load_config("midas")
    assert cfg["model"]["variant"] in {"dpt_large", "dpt_hybrid", "midas_small"}
    assert cfg["model"]["source"] == "hub"
    assert cfg["preprocessing"]["input_size"] == 384
    assert list(cfg["inference"]["clip_percentile"]) == [0.05, 0.995]
    assert cfg["eval"]["depth_cap_m"] == 80


def test_load_pipeline_config_values():
    cfg = load_config("pipeline")
    assert cfg["system"]["seed"] == 42
    assert all(isinstance(v, bool) for v in cfg["steps"].values())
    assert cfg["output"]["analysis_dir"] == "outputs/analysis"
    assert 11 in cfg["fusion"]["roi_classes"]
    assert len(cfg["difficulty"]["factor_weights"]) == 5


def test_pipeline_data_blocks_are_structured():
    cfg = load_config("pipeline")
    assert cfg["data"]["image_size"] == [512, 1024]
    assert cfg["data"]["cityscapes"]["root"] == "data/cityscapes"
    assert cfg["data"]["cityscapes"]["split"] == "train"
    assert cfg["data"]["cityscapes"]["max_samples"] is None
    assert cfg["data"]["kitti"]["root"] == "data/kitti"
    assert cfg["data"]["kitti"]["depth_cap_m"] == 80
    assert cfg["data"]["kitti"]["max_samples"] is None


def test_list_and_int_types():
    cfg = load_config("unet")
    assert all(isinstance(v, int) for v in cfg["data"]["image_size"])
    assert isinstance(cfg["train"]["epochs"], int)
    assert isinstance(cfg["train"]["learning_rate"], float)


def test_load_with_yaml_extension_equals_without():
    assert load_config("unet") == load_config("unet.yaml")


def test_missing_config_raises():
    with pytest.raises(ConfigError):
        load_config("does_not_exist")


def test_load_config_returns_copy():
    cfg = load_config("pipeline")
    cfg["system"]["seed"] = -1
    assert load_config("pipeline")["system"]["seed"] == 42


def test_overrides_merge_nested():
    base = load_config("unet")
    merged = load_config("unet", overrides={"train": {"epochs": 5}, "model": {"num_classes": 10}})
    assert merged["train"]["epochs"] == 5
    assert merged["model"]["num_classes"] == 10
    assert merged["env"]["seed"] == base["env"]["seed"]


def test_overrides_do_not_poison_cache():
    load_config("unet", overrides={"train": {"epochs": 5}})
    assert load_config("unet")["train"]["epochs"] == 60


def test_get_dotted_accessor():
    cfg = load_config("unet")
    assert get(cfg, "model.num_classes") == 19
    assert get(cfg, "missing.key", "fallback") == "fallback"
    assert get(cfg, "data.image_size") == [512, 1024]


def test_resolve_path_relative():
    assert resolve_path("outputs/analysis") == ROOT_DIR / "outputs" / "analysis"


def test_resolve_path_absolute_stays(tmp_path):
    assert resolve_path(tmp_path) == tmp_path.expanduser()


def test_resolve_path_creates_nothing():
    p = resolve_path("outputs/__never_created_by_config__")
    assert not p.exists()