"""Ensure the deterministic cuBLAS workspace config is set before any test can import
torch (01_CONVENTIONS.md §9). tasks.py sets this too for the `tasks.py test` entry point;
this covers bare `pytest` invocations."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
