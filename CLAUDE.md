# Helix 1.1 Factor Strategy Guidelines

## Commands
- **Run strategy**: `python3.11 helix_factor_strategy.py`
- **Run analysis**: `python3.11 analyze_strategy.py`
- **Lint**: `flake8 --max-line-length=100 *.py`
- **Type check**: `mypy --ignore-missing-imports *.py`

## Code Style
- **Imports**: Group standard library, third-party, local imports with blank line between
- **Formatting**: 4-space indentation, 100 character line length
- **Naming**:
  - Classes: PascalCase
  - Functions/methods: snake_case
  - Variables: snake_case
  - Constants: UPPER_SNAKE_CASE
- **Documentation**: Docstrings for all classes and functions
- **Error handling**: Use try/except blocks with specific exceptions
- **String formatting**: Use .format() method for Python compatibility (avoid f-strings)

## Strategy Components
- **SparseJumpModel**: Lightweight regime identification for factor ETFs
- **BlackLittermanOptimizer**: Portfolio optimization with long-only constraints
- **HelixFactorStrategy**: Main orchestrator for daily portfolio rebalancing

## Design Patterns
- Follow object-oriented principles with clear separation of concerns
- Use lightweight models avoiding transformers for computational efficiency
- Implement regime-aware allocation with jump penalty for persistence