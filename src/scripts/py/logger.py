import sys
from datetime import datetime


class Colors:
    RESET = "\033[0m"
    GRAY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"


class logger:
    @staticmethod
    def _timestamp():
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return f"{Colors.GRAY}{ts}{Colors.RESET}"

    @staticmethod
    def info(msg: str):
        print(f"{logger._timestamp()} {Colors.CYAN}[info]{Colors.RESET} {msg}")

    @staticmethod
    def warn(msg: str):
        print(f"{logger._timestamp()} {Colors.YELLOW}[warn]{Colors.RESET} {msg}")

    @staticmethod
    def error(msg: str):
        print(
            f"{logger._timestamp()} {Colors.RED}[error]{Colors.RESET} {msg}",
            file=sys.stderr,
        )

    @staticmethod
    def success(msg: str):
        print(f"{logger._timestamp()} {Colors.GREEN}[success]{Colors.RESET} {msg}")

    @staticmethod
    def debug(msg: str):
        print(f"{logger._timestamp()} {Colors.MAGENTA}[debug]{Colors.RESET} {msg}")
