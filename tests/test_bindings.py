# tests/test_bindings.py
import unittest
import os
import sys
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from python import chatglm

class TestBindings(unittest.TestCase):

    def test_bindings_import(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
