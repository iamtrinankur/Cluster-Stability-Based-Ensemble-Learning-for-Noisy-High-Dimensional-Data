# Contributing to NACC

Thank you for your interest in contributing to **Noise-Aware Consensus Clustering (NACC)**! This document provides guidelines for contributing code, benchmarks, documentation, or reporting issues.

---

## Development Setup

1. **Clone the repository with Git LFS**:
   `ash
   git clone https://github.com/iamtrinankur/Cluster-Stability-Based-Ensemble-Learning-for-Noisy-High-Dimensional-Data
   cd nacc-clustering
   git lfs pull
   `

2. **Create a virtual environment**:
   `ash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   `

3. **Install dependencies in editable mode**:
   `ash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .[dev]
   `

---

## Running Tests

Before submitting any changes, verify that the complete test suite passes:
`ash
python -m pytest tests/ -v
`

---

## Code Quality and Style

We follow PEP 8 and modern Python conventions.
- Format and lint with [Ruff](https://github.com/astral-sh/ruff):
  `ash
  ruff check .
  ruff format .
  `
- Write informative commit messages following conventional commits (eat:, ix:, docs:, 	est:, efactor:).

---

## Submitting Pull Requests

1. Fork the repository and create your feature branch:
   `ash
   git checkout -b feature/my-new-algorithm
   `
2. Commit your changes and ensure all tests pass.
3. Open a Pull Request against the main branch with a clear description of your contribution.