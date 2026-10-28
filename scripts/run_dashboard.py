#!/usr/bin/env python3
"""Launcher for the interactive TDLAS Methane Spectroscopy Cockpit.

Usage:
    python scripts/run_dashboard.py [--port 8050] [--host 127.0.0.1] [--no-browser]
"""

from __future__ import annotations
import argparse
import socket
import sys
import webbrowser
from functools import partial
from http.server import HTTPServer
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from tdlas.dashboard_backend import DashboardRequestHandler


def is_port_available(host: str, port: int) -> bool:
    """Test if a local TCP port is free to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(host: str, start_port: int, max_attempts: int = 50) -> int:
    """Find the first free TCP port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        if is_port_available(host, p):
            return p
    return start_port


def main():
    parser = argparse.ArgumentParser(description="Start TDLAS Interactive Dashboard Cockpit.")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind (default: 8050)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    args = parser.parse_args()

    port = args.port
    if not is_port_available(args.host, port):
        free_port = find_free_port(args.host, port + 1)
        print(f"Notice: Port {port} is occupied. Using port {free_port} instead.")
        port = free_port

    static_dir = project_root / "src" / "tdlas" / "static"
    handler_factory = partial(DashboardRequestHandler, directory=str(static_dir))
    
    server = HTTPServer((args.host, port), handler_factory)
    url = f"http://{args.host}:{port}/"

    print("==================================================================")
    print("     TDLAS METHANE SPECTROSCOPY ANALYSIS COCKPIT")
    print(f"     URL: {url}")
    print("     Press Ctrl+C to terminate.")
    print("==================================================================")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
