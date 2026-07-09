#!/usr/bin/env python3
"""
Shepherd Model Gateway CLI

Provides convenient command-line interface for launching the router and workers.

Usage:
    smg launch [args]          # Launch router only
    smg serve [args]           # Launch backend worker(s) + router
    smg --help                 # Show help
"""

from __future__ import annotations

import argparse
import os
import sys

from smg.smg_rs import (
    get_verbose_version_string,
    get_version_string,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the main CLI parser with subcommands."""
    prog_name = os.path.basename(sys.argv[0]) if sys.argv else "smg"
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description="Shepherd Model Gateway - High-performance inference router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Launch router subcommand
    subparsers.add_parser(
        "launch",
        help="Launch router only (requires existing worker URLs)",
        description="Launch the Shepherd router with existing worker instances",
        add_help=False,  # Let router handle --help
    )

    # Serve subcommand (two-pass parsing with lazy backend import)
    subparsers.add_parser(
        "serve",
        help="Launch backend worker(s) + gateway router",
        description="Launch inference backend workers and gateway router",
        add_help=False,  # Let serve handle --help with backend-specific args
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]

    # Handle version flags before parsing
    if argv and argv[0] in ["--version", "-V", "--version-verbose"]:
        if argv[0] == "--version-verbose":
            print(get_verbose_version_string())
        else:
            print(get_version_string())
        sys.exit(0)

    # Handle empty command - show help
    if not argv or argv[0] not in ["launch", "serve", "-h", "--help"]:
        parser = create_parser()
        parser.print_help()
        sys.exit(1)

    parser = create_parser()
    args, unknown = parser.parse_known_args(argv)

    if args.command == "launch":
        # Import and call launch_router functions directly
        from smg.launch_router import launch_router, parse_router_args

        # All router args are in unknown
        router_args = parse_router_args(unknown)
        launch_router(router_args)

    elif args.command == "serve":
        from smg.serve import serve_main

        serve_main(unknown)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
