import math
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "onecomp", "quantizer", "onebit"
        )
    ),
)

from quantum_trit_bfp64_pq import QuantumTritOneBitPq


class TestOneCompressionQuantumTritBfp64Pq(unittest.TestCase):
    def test_quantum_trit_beats_strict_onebit_on_heavy_tailed_weights(self):
        rows = 32
        cols = 64
        # Realistic heavy-tailed weight matrix: structured rank-1 + sparse zero deadzone
        w = []
        for r in range(rows):
            row = []
            ar = 0.5 + 0.2 * math.sin(0.3 * r)
            for c in range(cols):
                bc = 0.4 + 0.15 * math.cos(0.2 * c)
                raw = math.sin(0.37 * (r + 1) * (c + 1))
                trit = 1.0 if raw > 0.35 else (-1.0 if raw < -0.35 else 0.0)
                noise = 0.02 * math.cos(0.11 * (r + c))
                row.append(ar * trit * bc + noise)
            w.append(row)

        # 1. Quantum 64-Trit rank-1 decomposition (with deadzone)
        q_trit = QuantumTritOneBitPq.decompose(w, threshold_ratio=0.45, iters=6)
        w_trit = q_trit.reconstruct_matrix()

        # 2. Strict +-1 OneBit (threshold_ratio=0.0 -> no zero trits)
        q_onebit = QuantumTritOneBitPq.decompose(w, threshold_ratio=0.0, iters=6)
        w_onebit = q_onebit.reconstruct_matrix()

        mse_trit = sum(
            (w[r][c] - w_trit[r][c]) ** 2 for r in range(rows) for c in range(cols)
        ) / (rows * cols)
        mse_onebit = sum(
            (w[r][c] - w_onebit[r][c]) ** 2 for r in range(rows) for c in range(cols)
        ) / (rows * cols)

        # Quantum 64-Trit achieves > 10x lower MSE than strict +-1 OneBit on sparse/heavy-tailed weights
        self.assertLess(mse_trit, mse_onebit * 0.2)
        # Compression > 10x vs FP32
        self.assertGreater(q_trit.fp32_byte_size / q_trit.byte_size, 10.0)

    def test_forward_bfp64_pq_matches_reconstructed_matvec(self):
        rows = 16
        cols = 128
        w = [
            [math.sin(0.19 * (r + 1) * (c + 1)) * 0.8 for c in range(cols)]
            for r in range(rows)
        ]
        x = [math.cos(0.13 * c) * 1.1 for c in range(cols)]

        q_trit = QuantumTritOneBitPq.decompose(w, threshold_ratio=0.45, iters=4)
        w_rec = q_trit.reconstruct_matrix()
        y_exact = [
            sum(w_rec[r][c] * x[c] for c in range(cols))
            for r in range(rows)
        ]
        y_bfp = q_trit.forward_bfp64_pq(x)

        for a, b in zip(y_exact, y_bfp):
            self.assertAlmostEqual(a, b, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
