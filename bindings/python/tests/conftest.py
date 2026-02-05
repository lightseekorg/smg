"""
Pytest configuration for smg Python binding tests.

These are unit tests that run without GPU resources or external dependencies.
"""


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test (no GPU required)")
