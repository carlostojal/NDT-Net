
# Shared library integration tests

import unittest

class TestSharedLibraries(unittest.TestCase):

    def test_import_shared_library(self):
        import ctypes
        core = ctypes.cdll.LoadLibrary('core/build/libndtnetpp.so')
        self.assertTrue(core)

    def test_print_method_exists(self):
        import ctypes
        core = ctypes.cdll.LoadLibrary('core/build/libndtnetpp.so')
        self.assertTrue(hasattr(core, 'print_matrix'))

    def test_print_matrix(self):
        import ctypes
        import numpy as np
        core = ctypes.cdll.LoadLibrary('core/build/libndtnetpp.so')
        matrix = np.arange(9, dtype=np.float32).reshape(3, 3)
        rows, cols = matrix.shape
        matrix_ptr = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        core.print_matrix(matrix_ptr, rows, cols)
        self.assertTrue(True)
