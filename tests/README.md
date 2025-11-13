# SwiftLoader Rotation Tests

This directory contains tests for the sample index rotation functionality in SwiftLoader.

## Running Tests

To run all tests:

```bash
pytest tests/ -v
```

To run specific test files:

```bash
pytest tests/test_worker_rotation.py -v
pytest tests/test_rotation.py -v
```

To run tests for specific configurations:

```bash
pytest tests/test_worker_rotation.py::test_rotation_all_configurations -v
```

## Test Coverage

The tests verify:

1. **Rotation with shuffle=True**: Workers rotate through different sample ranges each epoch
2. **No rotation with shuffle=False**: Workers maintain fixed ranges across epochs
3. **All configurations**: Tests cover:
   - consumer=1, worker_per_consumer=1
   - consumer=1, worker_per_consumer=2
   - consumer=2, worker_per_consumer=2
4. **Coverage**: All samples in the dataset are covered across epochs
5. **No overlap**: Within an epoch, workers process non-overlapping ranges

## Test Files

- `test_worker_rotation.py`: Direct unit tests of Worker rotation logic
- `test_rotation.py`: Integration tests using SwiftLoader interface

