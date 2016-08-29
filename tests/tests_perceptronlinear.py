import os
import unittest
from thinkmachine import PerceptronLinear

class PerceptronLinearTest(unittest.TestCase):
	def setUp(self):
		self.perceptron = PerceptronLinear(learning_rate=0.5)
		self.dataset_default = [([1, 1], 	0),
								([1, 0], 	1),
								([0, 1], 	1),
								([0, 0], 	0)]

	def test_instance(self):
		self.assertIsInstance(self.perceptron, PerceptronLinear)

	def test_or(self):
		#	OR 		 inputs   expected
		dataset = [ ([1, 1], 	1),
					([1, 0], 	1),
					([0, 1], 	1),
					([0, 0], 	0)]

		self.perceptron.train(dataset)
		self.assertEqual(0, self.perceptron.perceive([0, 0]))
		

	def test_and(self):
		#	AND 		 inputs   expected
		dataset = [ ([1, 1], 	1),
					([1, 0], 	0),
					([0, 1], 	0),
					([0, 0], 	0)]

		self.perceptron.train(dataset)
		self.assertEqual(1, self.perceptron.perceive([1, 1]))

	def test_not(self):
		#	NOT 		 inputs   expected
		dataset = [ ([1], 	0),
					([0], 	1)]

		self.perceptron.train(dataset)
		self.assertEqual(0, self.perceptron.perceive([1]))


	def test_xor(self):
		#	XOR 		 inputs   expected
		dataset = [ ([1, 1], 	0),
					([1, 0], 	1),
					([0, 1], 	1),
					([0, 0], 	0)]

		self.perceptron.train(dataset)
		self.assertNotEqual(0, self.perceptron.perceive([1, 1]))


	def test_return_gerations(self):
		self.perceptron.train(self.dataset_default)
		self.assertEqual(10000, self.perceptron.generations)


	def test_return_learning_rate(self):
		self.perceptron.train(self.dataset_default)
		self.assertEqual(0.5, self.perceptron.learning_rate)

	def test_return_errors(self):
		#	OR 		 inputs   expected
		dataset = [( [1, 1], 	1),
					([1, 0], 	1),
					([0, 1], 	1),
					([0, 0], 	0)]

		self.perceptron.train(dataset)
		self.assertIsInstance(self.perceptron.errors, list)


if __name__ == '__main__':
    unittest.main()
