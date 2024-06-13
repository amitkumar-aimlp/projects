# Building a testcase with one unit test
# To build a test case, make a class that inherits from
# unittest.TestCase and write methods that begin with test_.
# Save this as test_full_names.py
# Testing middle names
# We’ve shown that get_full_name() works for first and last
# names. Let’s test that it works for middle names as well.
import unittest
from full_names_final import get_full_name

class NamesTestCase(unittest.TestCase):
    """Tests for names.py."""
    def test_first_last(self):
        """Test names like Janis Joplin."""
        full_name = get_full_name('janis', 'joplin')
        self.assertEqual(full_name, 'Janis Joplin')
    
    def test_middle(self):
        """Test names like David Lee Roth."""
        full_name = get_full_name('david', 'roth', 'lee')
        self.assertEqual(full_name, 'David Lee Roth')
    
# To run the test in Notebook-2
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)