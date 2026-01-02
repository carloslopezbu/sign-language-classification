import json

from rich.console import Console
from rich.table import Table

console = Console(
    log_time=True,
    log_time_format="%Y-%m-%d %H:%M:%S",
    log_path=False,
)


class logger:
    @staticmethod
    def info(msg: str) -> None:
        console.log(f"[blue](info)[/] {msg}")

    @staticmethod
    def warn(msg: str) -> None:
        console.log(f"[yellow](warn)[/] {msg}")

    @staticmethod
    def debug(msg: str) -> None:
        console.log(f"[magenta](debug)[/] {msg}")

    @staticmethod
    def error(msg: str) -> None:
        console.log(f"[red](error)[/] {msg}")

    @staticmethod
    def success(msg: str) -> None:
        console.log(f"[green](success)[/] {msg}")


def save_json(obj: object, dest: str):
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=3)


def load_json(src: str) -> dict:
    obj = {}
    with open(src, "r") as f:
        obj = json.load(f)
    return obj
