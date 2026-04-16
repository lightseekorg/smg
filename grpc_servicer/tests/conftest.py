"""Pytest configuration for smg-grpc-servicer unit tests.

These tests run without GPU resources or a live scheduler process.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (no GPU or scheduler required)"
    )
