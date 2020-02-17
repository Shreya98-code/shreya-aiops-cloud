import unittest
import class_example

class TestMyModule(unittest.TestCase):

	def setUp(self):
		return

	def test_do_divide(self):

		first_arg = 4
		second_arg = 2

		result = class_example.do_divide(first_arg, second_arg)

		expected_result = 3

		self.assertEqual(result, expected_result)