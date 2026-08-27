import pytest

from src.urdf_generator.samples import two_link_arm
from src.storage import config_store
from src.storage import database


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Isolates each test to its own temporary SQLite database file -
    replaces the old file-path monkeypatch (config_store no longer has
    a STORAGE_DIR attribute since the database rewrite)."""
    test_db_path = tmp_path / "test.db"
    test_engine = database.create_engine(
        f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}
    )
    test_session_local = database.sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    database.Base.metadata.create_all(bind=test_engine)

    monkeypatch.setattr(config_store, "SessionLocal", test_session_local)
    yield test_db_path


def test_save_and_load_round_trip():
    config = two_link_arm()
    result = config_store.save_design(config, "user-a")
    assert result["name"] == "two_link_arm"

    loaded = config_store.load_design(result["slug"], "user-a")
    assert loaded.name == config.name
    assert len(loaded.links) == len(config.links)


def test_list_designs_returns_saved_entries():
    config1 = two_link_arm()
    config1.name = "arm_alpha"
    config2 = two_link_arm()
    config2.name = "arm_beta"

    config_store.save_design(config1, "user-a")
    config_store.save_design(config2, "user-a")

    designs = config_store.list_designs("user-a")
    names = {d["name"] for d in designs}
    assert names == {"arm_alpha", "arm_beta"}


def test_saving_same_name_twice_overwrites():
    config = two_link_arm()
    config.name = "my_arm"
    config.payload_mass_kg = 0.5
    config_store.save_design(config, "user-a")

    config.payload_mass_kg = 99.0
    config_store.save_design(config, "user-a")

    designs = config_store.list_designs("user-a")
    assert len(designs) == 1
    loaded = config_store.load_design(designs[0]["slug"], "user-a")
    assert loaded.payload_mass_kg == 99.0


def test_loading_nonexistent_design_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="nonexistent_design"):
        config_store.load_design("nonexistent_design", "user-a")


def test_delete_removes_the_design():
    config = two_link_arm()
    result = config_store.save_design(config, "user-a")
    assert config_store.delete_design(result["slug"], "user-a") is True
    assert config_store.list_designs("user-a") == []
    assert config_store.delete_design(result["slug"], "user-a") is False


def test_path_traversal_attempt_is_sanitized():
    # The original file-based version tested that a malicious name like
    # "../../../etc/passwd" couldn't escape the storage directory via
    # path traversal - not applicable to a database (there's no
    # filesystem path derived from the name), but the same _slugify()
    # sanitization is still applied to the primary key, so this checks
    # the slug itself is safe rather than a file location.
    config = two_link_arm()
    config.name = "../../../etc/passwd"
    result = config_store.save_design(config, "user-a")

    assert ".." not in result["slug"]
    assert "/" not in result["slug"]

    designs = config_store.list_designs("user-a")
    assert len(designs) == 1
