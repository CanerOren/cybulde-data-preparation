import logging
import socket
import subprocess


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"[{socket.gethostname()}] {name}")


def run_shell_command(cmd: str) -> str:
    p = subprocess.run(cmd, text=True, shell=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n" f"STDOUT:\n{p.stdout}\n" f"STDERR:\n{p.stderr}")
    return p.stdout