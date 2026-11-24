"""One entrypoint, portable environment loading and IPv6 container binding."""

import os
from pathlib import Path
import socket

import pytest
from sqlalchemy.engine import make_url

from src import cli


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    for name in ("DATABASE_URL", "DATABASE_HOST", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "TRAINING_BACKEND"):
        # Track absent keys too, so direct dotenv writes are undone after the test.
        monkeypatch.setenv(name, os.environ.get(name, ""))
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_dotenv_preserves_exported_configuration(settings, monkeypatch):
    (settings / ".env").write_text("DATABASE_URL=postgresql+psycopg://from-file/db\n")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://exported/db")
    cli.load_environment()
    assert os.environ["DATABASE_URL"] == "postgresql+psycopg://exported/db"


def test_container_database_host_and_password_encoding(settings, monkeypatch):
    password = "p@$$:#/word"
    monkeypatch.setenv("DATABASE_HOST", "db")
    monkeypatch.setenv("POSTGRES_USER", "pix")
    monkeypatch.setenv("POSTGRES_PASSWORD", password)
    monkeypatch.setenv("POSTGRES_DB", "research")
    cli.load_environment()
    url = make_url(os.environ["DATABASE_URL"])
    assert (url.host, url.port, url.username, url.database, url.password) == ("db", 5432, "pix", "research", password)


def test_train_defaults_to_gpu(settings, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "train", lambda *args: calls.append(args) or Path("run"))
    cli.main(["train", "--scenario", "10k"])
    assert len(calls) == 1 and calls[0][:3] == ("10k", None, "gpu")


def test_serve_defaults_to_local_ipv6(settings, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "serve", lambda *args: calls.append(args))
    cli.main(["serve"])
    assert calls == [("::1", 8000)]


def test_serve_rejects_invalid_port(settings):
    with pytest.raises(SystemExit) as error:
        cli.main(["serve", "--port", "0"])
    assert error.value.code == 2


def test_container_ipv6_listener_accepts_mapped_ipv4(monkeypatch):
    import uvicorn
    observed = []

    class InspectServer:
        def __init__(self, config):
            self.config = config

        def run(self, sockets):
            listener = sockets[0]
            assert listener.family == socket.AF_INET6
            assert listener.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 0
            with socket.create_connection(("127.0.0.1", listener.getsockname()[1]), timeout=2):
                connection, address = listener.accept()
                connection.close()
                observed.append(address[0])

    monkeypatch.setattr(uvicorn, "Server", InspectServer)
    cli.serve("::", 0)
    assert observed == ["::ffff:127.0.0.1"]


def test_migrate_uses_canonical_alembic_configuration(settings, monkeypatch):
    from alembic import command
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://example/db")
    calls = []
    monkeypatch.setattr(command, "upgrade", lambda config, revision: calls.append((config.config_file_name, revision)))
    cli.main(["migrate"])
    assert calls == [(str(settings / "alembic.ini"), "head")]
