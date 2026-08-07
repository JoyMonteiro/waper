# Phase 0 & Phase 2 Execution Plan

This document outlines the specific steps required to complete the remaining tasks of Phase 0 and the entirety of Phase 2 as specified in the `waper_refactoring_spec.md`.

## Phase 0: Final Cleanup

### Step 1: Remove Remaining Dead Code (Task 0.2)
- **Target Files:** `waper/identification/rwp_graph.py`, `waper/identification/topology.py`, `waper/tracking/tracking_graph.py`.
- **Action:** Delete all commented-out function bodies, logic blocks, and `# print()` statements identified in the refactoring spec.
- **Verification:** `grep -r "^ *#.*" waper/` shows only meaningful documentation comments.

### Step 2: Final Logging Sweep (Task 0.4)
- **Target Files:** All files in `waper/`.
- **Action:** Convert any remaining active `print()` calls into `logger` calls. Ensure `logging.basicConfig` is correctly toggled by the `debug` flag in `Waper.__init__`.
- **Verification:** `grep -r "print(" waper/` returns zero active results.

---

## Phase 2: Critical Bug Fixes

### Step 3: Utility & Error Handling (Tasks 2.3 & 2.6)
- **Task 2.3:** Refactor `is_to_the_east` in `utils.py`.
    - Rename variable to `delta_lon`.
    - Ensure strict boolean return.
    - **New File:** Create `tests/test_utils.py` with the 5 specified test cases.
- **Task 2.6:** Fix bare `except:` in `rwp_polygon.py`.
    - Replace with `except Exception as e:`.
    - Include full error message in `logger.error` and `ValueError`.
- **Verification:** `pytest tests/test_utils.py` passes.

### Step 4: ID Collision Refactor (Tasks 2.2 & 2.5)
- **Task 2.2 (Node IDs):** Transition `rwp_graph.py` to tuple-based node IDs `("max", id)` and `("min", id)`.
    - Remove the `min_id = 100` remapping hack.
    - Update all downstream logic (e.g., `visualization.py`, `rwp_polygon.py`) that checks the sign of node IDs.
- **Task 2.5 (Feature IDs):** Implement unique monotonic integer IDs for RWPs in `api.py`.
    - Remove `round(path_max, 2)` dependency in `rwp_polygon.py`.
    - Update `tracking_graph.py` to use exact integer comparison for feature matching.
- **Verification:** `pytest tests/test_association_graph.py` and integration tests pass.

### Step 5: Geometry & Performance (Task 2.4)
- **Task 2.4:** Vectorize association graph construction in `rwp_graph.py`.
    - Use `scipy.spatial.cKDTree` for nearest-neighbor lookups.
    - Use full 3D Cartesian coordinates (`mesh.points`) for spherical accuracy.
    - Eliminate the O(C * M) and O(C * N) nested loops.
- **Verification:** `test_date_line_association` passes. Benchmarking shows significant speedup on high-resolution data.

---

## Definition of Done for Phase 2
- [x] No node or feature ID collisions occur during tracking.
- [x] Spherical distance logic is correct at the poles.
- [x] No bare `except` blocks exist in the source.
- [x] `is_to_the_east` is fully tested and robust.
- [x] Integration tests pass across all synthetic fixtures.
