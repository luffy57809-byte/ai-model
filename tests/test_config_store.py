import os
import pytest
from src.urdf_generator.samples import two_link_arm
from src.storage import config_store


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    test_storage_dir = tmp_path / "saved_designs"
    monkeypatch.setattr(config_store, "STORAGE_DIR", test_storage_dir)
    yield test_storage_dir


def test_save_and_load_round_trip():
    config = two_link_arm()
    result = config_store.save_design(config)
    assert result["name"] == "two_link_arm"
    loaded = config_store.load_design(result["slug"])
    assert loaded.name == config.name
    assert len(loaded.links) == len(config.links)


def test_list_designs_returns_saved_entries():
    config1 = two_link_arm()
    config1.name = "arm_alpha"
    config2 = two_link_arm()
    config2.name = "arm_beta"
    config_store.save_design(config1)
    config_store.save_design(config2)
    designs = config_store.list_designs()
    names = {d["name"] for d in designs}
    assert names == {"arm_alpha", "arm_beta"}


def test_saving_same_name_twice_overwrites():
    config = two_link_arm()
    config.name = "my_arm"
    config.payload_mass_kg = 0.5
    config_store.save_design(config)
    config.payload_mass_kg = 99.0
    config_store.save_design(config)
    designs = config_store.list_designs()
    assert len(designs) == 1
    loaded = config_store.load_design(designs[0]["slug"])
    assert loaded.payload_mass_kg == 99.0


def test_loading_nonexistent_design_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="nonexistent_design"):
        config_store.load_design("nonexistent_design")


def test_delete_removes_the_design():
    config = two_link_arm()
    result = config_store.save_design(config)
    assert config_store.delete_design(result["slug"]) is True
    assert config_store.list_designs() == []
    assert config_store.delete_design(result["slug"]) is False


def test_path_traversal_attempt_is_sanitized(isolated_storage):
    config = two_link_arm()
    config.name = "../../../etc/passwd"
    result = config_store.save_design(config)
    saved_files = list(isolated_storage.glob("*.json"))
    assert len(saved_files) == 1
    assert saved_files[0].parent == isolated_storage
    assert ".." not in result["slug"]
    assert "/" not in result["slug"]
