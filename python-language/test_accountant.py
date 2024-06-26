# Using setUp() to support multiple tests
# The instance self.acc can be used in each new test.
import unittest
from accountant import Accountant

class TestAccountant(unittest.TestCase):
    """Tests for the class Accountant."""
    
    def setUp(self):
        self.acc = Accountant()
    
    def test_initial_balance(self):
        # Default balance should be 0.
        self.assertEqual(self.acc.balance, 0)
        # Test non-default balance.
        acc = Accountant(100)
        self.assertEqual(acc.balance, 100)
    
    def test_deposit(self):
        # Test single deposit.
        self.acc.deposit(100)
        self.assertEqual(self.acc.balance, 100)
        # Test multiple deposits.
        self.acc.deposit(100)
        self.acc.deposit(100)
        self.assertEqual(self.acc.balance, 300)
    
    def test_withdrawal(self):
        # Test single withdrawal.
        self.acc.deposit(1000)
        self.acc.withdraw(100)
        self.assertEqual(self.acc.balance, 900)
        
# To run the test in Notebook-2
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)