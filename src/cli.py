"""Single execution interface for both scenarios and legacy commands."""

import argparse
import json
import os
from pathlib import Path
import socket

from dotenv import load_dotenv
from sqlalchemy.engine import URL

from src.workflow import GPU_RESULTS, ROOT, evaluate, report, train, verify
from src.utils.cuda import GPUExecutionError


def active_run(root: Path, scenario: str) -> Path:
    """Resolve only a trusted local active pointer within its scenario directory."""
    pointer_path = root / scenario / "active.json"
    if not pointer_path.is_file():
        raise FileNotFoundError(
            f"Cenário {scenario} sem execução ativa em {root}. "
            f"Conclua 'python main.py train --scenario {scenario}' neste ambiente, "
            "ou informe --results/--run para consultar uma execução existente."
        )
    pointer = json.loads(pointer_path.read_text())
    run = root / scenario / pointer["run_id"]
    if run.parent.resolve() != (root / scenario).resolve():
        raise ValueError("Identificador de execução inválido")
    return run


def load_environment() -> None:
    """Load local settings without overriding exported values; Compose owns its DB host."""
    load_dotenv(ROOT / ".env", override=False)
    if os.environ.get("DATABASE_HOST"):
        url = URL.create("postgresql+psycopg", host=os.environ["DATABASE_HOST"],
                         username=os.environ.get("POSTGRES_USER", "pix"),
                         password=os.environ.get("POSTGRES_PASSWORD", "pix"),
                         database=os.environ.get("POSTGRES_DB", "pix"), port=5432)
        os.environ["DATABASE_URL"] = url.render_as_string(hide_password=False)


def serve(host: str, port: int) -> None:
    """Serve locally by default; a container wildcard uses one native dual-stack socket."""
    import uvicorn
    if host == "::":
        with socket.create_server((host, port), family=socket.AF_INET6, dualstack_ipv6=True) as listener:
            server = uvicorn.Server(uvicorn.Config("src.api.app:app", host=host, port=port))
            server.run(sockets=[listener])
    else:
        uvicorn.run("src.api.app:app", host=host, port=port)


def main(argv: list[str] | None = None) -> None:
    """Training requires an explicit command; imports have no training side effects."""
    load_environment()
    parser = argparse.ArgumentParser(description="Treinamento, avaliação, histórico e API Pix")
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve", help="Iniciar a API com os modelos finais")
    server.add_argument("--host", default="::1", help="Endereço local; use :: dentro do contêiner")
    server.add_argument("--port", type=int, default=8000)
    commands.add_parser("migrate", help="Aplicar as migrações pendentes do PostgreSQL")
    for command in ("train", "evaluate", "report", "verify", "import-history"):
        child = commands.add_parser(command)
        child.add_argument("--scenario", choices=("10k", "2m", "all"), default="2m")
        child.add_argument("--results", type=Path, default=None,
                           help="Diretório de resultados; padrão: runs-gpu (GPU), runs (treino CPU explícito)")
        if command in {"train", "import-history"}:
            child.add_argument("--source", type=Path, help="CSV local (10k com ;), ou parquet")
        else:
            child.add_argument("--run", type=Path, help="Execução congelada; padrão: ativa do cenário")
        if command == "train":
            child.add_argument("--backend", choices=("cpu", "gpu"), default=os.environ.get("TRAINING_BACKEND", "gpu"))
            child.add_argument("--tuning-rows", type=int, default=80000,
                               help="Janela final do desenvolvimento para busca; 0 usa todo desenvolvimento")
        if command == "import-history":
            child.add_argument("--cutoff", required=True, help="Importa somente pagamentos anteriores a este instante")
    args = parser.parse_args(argv)
    if args.command == "serve":
        if not 1 <= args.port <= 65535:
            parser.error("--port deve estar entre 1 e 65535")
        serve(args.host, args.port)
        return
    if args.command == "migrate":
        if not os.environ.get("DATABASE_URL"):
            parser.error("Configure DATABASE_URL no .env antes de aplicar as migrações")
        from alembic import command
        from alembic.config import Config
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        return
    if args.scenario == "all" and (getattr(args, "source", None) or getattr(args, "run", None)):
        parser.error("--source e --run exigem um cenário específico")
    for scenario in (("10k", "2m") if args.scenario == "all" else (args.scenario,)):
        if args.command == "train":
            try:
                output = train(scenario, args.source, args.backend, args.tuning_rows, args.results)
            except GPUExecutionError as exc:
                parser.error(str(exc))
            print(output)
        elif args.command == "import-history":
            from src.api.history_import import import_history
            print(json.dumps(import_history(scenario, args.cutoff, args.source), ensure_ascii=False))
        else:
            try:
                run = args.run or active_run(args.results or GPU_RESULTS, scenario)
            except FileNotFoundError as exc:
                parser.error(str(exc))
            operation = {"evaluate": evaluate, "report": report, "verify": verify}[args.command]
            print(operation(run))
