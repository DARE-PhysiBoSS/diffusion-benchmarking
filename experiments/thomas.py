import numpy as np

import unittest


def thomas_algorithm(a, b, c, d):
    """
    Solves a tridiagonal system of equations using the Thomas algorithm.

    Parameters:
        a (list): Sub-diagonal coefficients (a[0] is unused).
        b (list): Main diagonal coefficients.
        c (list): Super-diagonal coefficients (c[-1] is unused).
        d (list): Right-hand side values.

    Returns:
        list: Solution vector.
    """

    a = a[:]
    b = b[:]
    c = c[:]
    d = d[:]

    n = len(d)
    # Forward elimination
    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    # Back substitution
    x = [0] * n
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def cyclic_reduction(a, b, c, d):
    """
    Solves a tridiagonal system of equations using the Cyclic Reduction algorithm.

    Parameters:
        a (list): Sub-diagonal coefficients (a[0] is unused).
        b (list): Main diagonal coefficients.
        c (list): Super-diagonal coefficients (c[-1] is unused).
        d (list): Right-hand side values.

    Returns:
        list: Solution vector.
    """
    n = len(d)
    log_n = int(np.log2(n))

    a = a[:]
    b = b[:]
    c = c[:]
    d = d[:]

    for step in range(log_n):
        stride = 1 << step
        tmp = -1
        for i in range(2 * stride - 1, n - 2 * stride, 2 * stride):
            alpha = a[i] / b[i - stride]
            beta = c[i] / b[i + stride]

            b[i] -= alpha * c[i - stride] + beta * a[i + stride]
            d[i] -= alpha * d[i - stride] + beta * d[i + stride]

            a[i] = -alpha * a[i - stride]
            c[i] = -beta * c[i + stride]
            tmp = i

            # print(f"A n: {n}, i: {i}, stride: {stride}")

        for i in range(tmp + 2 * stride, n, 2 * stride):

            # print(f"B n: {n}, i: {i}, stride: {stride}")
            if i + stride >= n:
                alpha = a[i] / b[i - stride]

                b[i] -= alpha * c[i - stride]
                d[i] -= alpha * d[i - stride]

                a[i] = -alpha * a[i - stride]
            else:
                alpha = a[i] / b[i - stride]
                beta = c[i] / b[i + stride]

                b[i] -= alpha * c[i - stride] + beta * a[i + stride]
                d[i] -= alpha * d[i - stride] + beta * d[i + stride]

                a[i] = -alpha * a[i - stride]
                c[i] = -beta * c[i + stride]

    x = d.copy()
    x[2**log_n - 1] = x[2**log_n - 1] / b[2**log_n - 1]

    for step in range(log_n - 1, -1, -1):
        stride = 1 << step
        
        for i in range(stride - 1, stride, 2 * stride):
            x[i] = (x[i] - c[i] * x[i + stride]) / b[i]
        
        for i in range(stride - 1 + 2 * stride, n, 2 * stride):
            if i + stride >= n:
                x[i] = (x[i] - a[i] * x[i - stride]) / b[i]
            else:
                x[i] = (x[i] - a[i] * x[i - stride] - c[i] * x[i + stride]) / b[i]

    return x


def parallel_cyclic_reduction(a, b, c, d):
    """
    Solves a tridiagonal system of equations using the Parallel Cyclic Reduction algorithm.

    Parameters:
        a (list): Sub-diagonal coefficients (a[0] is unused).
        b (list): Main diagonal coefficients.
        c (list): Super-diagonal coefficients (c[-1] is unused).
        d (list): Right-hand side values.

    Returns:
        list: Solution vector.
    """

    n = len(d)
    log_n = 0
    while (1 << log_n) < n:
        log_n += 1

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)

    for step in range(log_n):
        stride = 1 << step
        equations = (step + 1)
        for eq_i in range(equations):
            eq_n = n // equations
            
            for i in range(eq_n):

                idx = eq_i + i * stride

                if idx + stride >= n:
                    alpha = a[idx] / b[idx - stride]
                    b[idx] -= alpha * c[idx - stride]
                    d[idx] -= alpha * d[idx - stride]
                    a[idx] = -alpha * a[idx - stride]
                else:
                    alpha = a[idx] / b[idx - stride]
                    beta = c[idx] / b[idx + stride]
                    b[idx] -= alpha * c[idx - stride] + beta * a[idx + stride]
                    d[idx] -= alpha * d[idx - stride] + beta * d[idx + stride]
                    a[idx] = -alpha * a[idx - stride]
                    c[idx] = -beta * c[idx + stride]


    return x.tolist()


class TestTridiagonalSolvers(unittest.TestCase):
    def setUp(self):
        # Example tridiagonal system of size n
        self.n = 4  # You can change this value for different sizes
        self.a = [0] + [1] * (self.n - 1)  # Sub-diagonal (a[0] is unused)
        self.b = [4] * self.n  # Main diagonal
        self.c = [1] * (self.n - 1) + [0]  # Super-diagonal (c[-1] is unused)
        self.d = [5] + [6] * (self.n - 2) + [5]  # Right-hand side

    def test_thomas_algorithm(self):
        # Expected solution calculated manually for n=10 is [1, 1, ..., 1]
        expected_solution = [1.] * self.n
        result = thomas_algorithm(self.a, self.b, self.c, self.d)
        for r, e in zip(result, expected_solution):
            self.assertAlmostEqual(r, e, places=7)

    def test_cyclic_reduction(self):
        thomas_result = thomas_algorithm(self.a, self.b, self.c, self.d)
        cyclic_result = cyclic_reduction(self.a, self.b, self.c, self.d)
        for t, c in zip(thomas_result, cyclic_result):
            self.assertAlmostEqual(t, c, places=7)

    # def test_for_various_sizes(self):
    #     for _ in range(1000):
    #         for n in [5, 10, 12, 20, 50, 8, 16, 32, 64, 128, 127, 129]:
    #             a = [0] + list(np.random.randint(1, 5, size=n - 1))
    #             b = list(np.random.randint(10, 20, size=n))
    #             c = list(np.random.randint(1, 5, size=n - 1)) + [0]
    #             d = list(np.random.randint(1, 10, size=n))

    #             thomas_result = thomas_algorithm(a, b, c, d)
    #             cyclic_result = cyclic_reduction(a, b, c, d)

    #             for t, c in zip(thomas_result, cyclic_result):
    #                 self.assertAlmostEqual(t, c, places=7)

    # def test_parallel_cyclic_reduction(self):
    #     thomas_result = thomas_algorithm(self.a, self.b, self.c, self.d)
    #     parallel_result = parallel_cyclic_reduction(
    #         self.a, self.b, self.c, self.d)
    #     for t, p in zip(thomas_result, parallel_result):
    #         self.assertAlmostEqual(t, p, places=7)


if __name__ == "__main__":
    unittest.main()
