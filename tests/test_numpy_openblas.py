"""Tests for the local OpenBLAS-linked NumPy build."""

from __future__ import annotations

import io
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout

import numpy as np


def blas_backend_name() -> str:
    config = np.__config__.show(mode="dicts")
    return str(config["Build Dependencies"]["blas"]["name"]).lower()


class TestNumpyOpenBLASBuild(unittest.TestCase):
    """Verify NumPy was built against OpenBLAS and basic linear algebra works."""

    def test_blas_backend_is_openblas(self) -> None:
        self.assertEqual(
            blas_backend_name(),
            "openblas",
            "NumPy must be built with OpenBLAS for multithreaded PIC workloads.",
        )

    def test_lapack_backend_is_openblas(self) -> None:
        config = np.__config__.show(mode="dicts")
        lapack = str(config["Build Dependencies"]["lapack"]["name"]).lower()
        self.assertEqual(lapack, "openblas")

    def test_matrix_multiply(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((512, 512))
        b = rng.standard_normal((512, 512))
        expected = np.dot(a, b)
        np.testing.assert_allclose(a @ b, expected, rtol=1e-10, atol=1e-10)

    def test_linear_solve(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.standard_normal((64, 64))
        a = a @ a.T + 64.0 * np.eye(64)
        x = rng.standard_normal(64)
        b = a @ x
        solved = np.linalg.solve(a, b)
        np.testing.assert_allclose(solved, x, rtol=1e-9, atol=1e-9)


class TestNumpyOpenBLASMultithreading(unittest.TestCase):
    """Exercise OpenBLAS threading via OPENBLAS_NUM_THREADS."""

    @staticmethod
    def _timed_matmul(size: int = 1024, repeats: int = 4) -> float:
        rng = np.random.default_rng(42)
        a = rng.standard_normal((size, size))
        b = rng.standard_normal((size, size))

        # Warm up BLAS thread pool before timing.
        _ = a @ b

        start = time.perf_counter()
        for _ in range(repeats):
            _ = a @ b
        return time.perf_counter() - start

    def test_concurrent_matmul_results_are_finite(self) -> None:
        size = 128

        def worker(seed: int) -> np.ndarray:
            local_rng = np.random.default_rng(seed)
            x = local_rng.standard_normal((size, size))
            y = local_rng.standard_normal((size, size))
            return x @ y

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(worker, range(16)))

        for result in results:
            self.assertEqual(result.shape, (size, size))
            self.assertTrue(np.isfinite(result).all())

    def test_openblas_thread_count_changes_performance(self) -> None:
        """More OpenBLAS threads should not be slower on a large GEMM."""
        if (os.cpu_count() or 1) < 2:
            self.skipTest("Need at least 2 CPUs to compare OpenBLAS thread counts.")

        previous = os.environ.get("OPENBLAS_NUM_THREADS")
        try:
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            single_thread = self._timed_matmul()

            os.environ["OPENBLAS_NUM_THREADS"] = "4"
            multi_thread = self._timed_matmul()
        finally:
            if previous is None:
                os.environ.pop("OPENBLAS_NUM_THREADS", None)
            else:
                os.environ["OPENBLAS_NUM_THREADS"] = previous

        self.assertLess(
            multi_thread,
            single_thread * 1.5,
            "OPENBLAS_NUM_THREADS=4 should not be materially slower than 1 thread.",
        )


if __name__ == "__main__":
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        np.show_config()
    print(buffer.getvalue())
    unittest.main()
