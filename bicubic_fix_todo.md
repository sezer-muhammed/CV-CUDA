# Bicubic Interpolation Boundary TODO

## Public Evidence & Current Status
- The repository README already flags the issue: *“The Resize and RandomResizedCrop operators incorrectly interpolate pixel values near the boundary of an image or tensor when using cubic interpolation.”* (`README.md:67`). No fix is merged yet.
- Users report visible bands or flattened edges when calling any front-end that forwards to these operators with `Interp.CUBIC` (Python, C++, Torch bindings, etc.).

## Impacted APIs And Code Paths

| Public entry point | Binding file | C++ operator | Legacy implementation | Notes |
| --- | --- | --- | --- | --- |
| `cvcuda.resize` / `resize_into` (tensor & var-shape) | `python/mod_cvcuda/OpResize.cpp:32-206` | `cvcuda::Resize` (`src/cvcuda/priv/OpResize.cpp:24-63`) | `legacy::ResizeVarShape::infer` (`src/cvcuda/priv/legacy/resize_var_shape.cu:431-470`) which launches `resize_bicubic` (`:164-229`) | This is the path used by most samples/tests. |
| `cvcuda.random_resized_crop` / `_into` | `python/mod_cvcuda/OpRandomResizedCrop.cpp:32-213` | `cvcuda::RandomResizedCrop` (`src/cvcuda/priv/OpRandomResizedCrop.cpp:24-63`) | `legacy::RandomResizedCrop` + `RandomResizedCropVarShape` (tensor kernel in `src/cvcuda/priv/legacy/random_resized_crop.cu:180-223`, var-shape version in `random_resized_crop_var_shape.cu:185-231`) | Same bicubic math is duplicated here. |
| `cvcuda.pillowresize` | `python/mod_cvcuda/OpPillowResize.cpp:43-229` | `cvcuda::PillowResize` (`src/cvcuda/priv/OpPillowResize.cpp:31-91`) | Uses `legacy::PillowResize` (`src/cvcuda/priv/legacy/pillow_resize*.cu`). These kernels have their own filter precomputation and are not part of the README warning, but they should be re-tested once the shared fix is ready. |

> **Test coverage gap:** `tests/cvcuda/python/test_opresize.py:51-226` and `tests/cvcuda/python/test_oprandomresizedcrop.py:51-145` only assert shape/layout/dtype and never validate pixel correctness at the borders, so the regression slipped through.

## Root Cause (Legacy Bicubic Kernels)

### `resize_var_shape.cu`
- In `resize_bicubic` the source coordinate is computed in floating-point (`fy`/`fx`), then the integer anchor is clamped into `[1, H-3]` or `[1, W-3]` (`src/cvcuda/priv/legacy/resize_var_shape.cu:186-205`). The fractional offsets are left as if the coordinate had *not* been clamped. When the original sample location lies outside the safe region, the kernel still reads a clamped 4×4 neighborhood but applies weights derived from the unclamped offsets, producing biased contributions along the top/bottom edges.
- Horizontally the code multiplies `fx` by `((sx >= 1) && (sx < width - 3))` (`:201-205`). Effectively `fx` becomes `0` when the unclamped coordinate is out of range, collapsing the cubic polynomial to the center tap and causing “flattened” edges.
- The same math exists in the var-shape specialization and both are guarded by `LEGACY_BICUBIC_MATH_VS`, which optionally applies `abs()` to the accumulated value. That conditional does not fix the weighting error.

### `random_resized_crop.cu` and `_var_shape.cu`
- The bicubic branch duplicates the exact same logic, with the only difference being the extra crop offsets (`top/left`). See `src/cvcuda/priv/legacy/random_resized_crop.cu:180-205` and `random_resized_crop_var_shape.cu:185-210`. Consequently RandomResizedCrop shows the same boundary artifacts whenever the chosen crop touches the image edges (a common case when scale≈1).

### Why PillowResize is likely unaffected
- `legacy::PillowResize` precomputes filter taps per output pixel (`pillow_resize_var_shape.cu:41-150`) and keeps consistent bounds/weights, so it does not share this particular bug. Earlier reports conflated PillowResize with Resize because both expose bicubic interpolation to Python; this TODO focuses on the README-confirmed `Resize`/`RandomResizedCrop` paths.

## Fix Plan
1. **Recompute fractional offsets after clamping**
   - Store the unclamped integer coordinate before applying `cuda::max/min`.
   - After clamping, add `(original_s - clamped_s)` back into `fx/fy` (or equivalently clamp using border accessors so that fractional offsets remain untouched).
   - Apply this to both axes in:
     - `src/cvcuda/priv/legacy/resize_var_shape.cu:186-205` (tensor + var-shape kernel).
     - `src/cvcuda/priv/legacy/random_resized_crop.cu:180-205`.
     - `src/cvcuda/priv/legacy/random_resized_crop_var_shape.cu:185-210`.
2. **Consider border wrappers instead of manual clamping**
   - `cuda::BorderVarShapeWrap` is already used for AREA interpolation (`resize_var_shape.cu:465-468`). Extending bicubic to read through a border wrapper simplifies the math and avoids manual conditionals.
3. **Audit other consumers**
   - Search for `LEGACY_BICUBIC_MATH` and `fx *= (` patterns to ensure no other kernels copy this bug.
4. **Remove the `fx *= condition` workaround**
   - Once rate-limited clamping is correct, that line should be deleted; it currently hides the bug by forcing `fx=0`.
5. **Optional refactor**
   - Extract shared bicubic weight computation into a helper to keep Resize and RandomResizedCrop in sync and make future maintenance easier.

## Validation Plan
1. **Unit tests (Python)**
   - Extend `tests/cvcuda/python/test_opresize.py` with a bicubic testcase that resizes a horizontal gradient tensor, compares against a CPU reference (e.g., OpenCV or Pillow), and asserts per-pixel error near the first/last 4 rows/columns.
   - Do the same for `tests/cvcuda/python/test_oprandomresizedcrop.py`, forcing crop parameters that align the output border with the source border.
2. **Unit tests (C++)**
   - Add a small golden test under `tests/cvcuda/cpp` (or similar) that feeds a known 5×5 image through `cvcuda::Resize` and checks the CUDA output against a precomputed host result.
3. **Visual sanity**
   - Reproduce the before/after behavior using a script that calls `cvcuda.resize` on a checkerboard and dumps the first/last rows to confirm the band disappears.
4. **Performance regression check**
   - Run existing resize/random-resized-crop benchmarks (`bench/BenchResize*.cpp`, `bench/BenchRandomResizedCrop.cpp`) to make sure the extra math does not degrade throughput noticeably.
5. **Documentation**
   - Once validated, remove or update the warning in `README.md:67` and mention the fix in the release notes.

## TODO Checklist
- [ ] Patch `resize_bicubic` (tensor + var-shape paths) to rebase fractional offsets after clamping.
- [ ] Patch `random_resized_crop` (tensor + var-shape) to use the corrected math.
- [ ] Add regression tests that compare against a CPU bicubic reference and fail if boundary pixels deviate beyond a tolerance.
- [ ] Verify PillowResize to ensure no regressions and document if it shares or does not share the issue.
- [ ] Update README/release notes once the bug is fixed.
