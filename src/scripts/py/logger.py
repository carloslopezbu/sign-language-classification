from rich.console import Console

console = Console(log_time=True, log_time_format="%Y-%m-%d %H:%M:%S", log_path=False)


def info(msg: str) -> None:
    console.log(f"[blue](info)[/] {msg}")


def warn(msg: str) -> None:
    console.log(f"[yellow](warn)[/] {msg}")


def debug(msg: str) -> None:
    console.log(f"[magenta](debug)[/] {msg}")


def error(msg: str) -> None:
    console.log(f"[red](error)[/] {msg}")


def success(msg: str) -> None:
    console.log(f"[green](success)[/] {msg}")
