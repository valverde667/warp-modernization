# Warp Modernization for Python 3.11 + NumPy 2.0 + macOS ARM (PyGist/CGM Working)

This branch updates Warp to run successfully in modern environments (tested on macOS ARM, Python 3.11, NumPy 2.0) while preserving PyGist/CGM plotting functionality.

## Summary of Changes
- **Replaced deprecated `alltrue` calls** with `numpy.all` equivalents.
- **Fixed `min`/`max` reductions** in `genericpf` by switching to `np.minimum` / `np.maximum` lambdas, ensuring correct elementwise behavior.
- **Patched `arraysliceoperation`** to ensure scalar results are wrapped as arrays before reductions, preventing type and shape errors in recursive calls.
- **Updated `np.array(..., copy=False)` calls** to `np.asarray(...)` for NumPy 2.0 compatibility (avoids new `ValueError` when copy is unavoidable).
- Minor adjustments to support modern NumPy dtypes and stricter broadcasting rules.

## Usage Notes
After pulling these changes, **you must run**:
```
make install
'''
inside the `pywarp90' directory to rebuild Warp with the updated source. 

This step is required after each change to ensure the compiled Python modules reflect 
the updated source code. 

## Impact
- Fully functional Warp simulations on Python 3.11 + NumPy 2.0.

- Restored plotting routines using CGM/PyGist.

- Tested on macOS ARM (Apple Silicon) — all example runs and beam plotting routines execute without errors.