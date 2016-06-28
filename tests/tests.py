import os
import unittest
from thinkmachine import ID3

class ID3Test(unittest.TestCase):
	def setUp(self):
		self.data = []
		self.id3 = ID3()
		self.tree = None

		self.id3.load_sample()

		file = open('data_set/sample.csv')

		for line in file:
			line = line.strip("\r\n")
			self.data.append(line.split(','))

		attributes = self.data[0]
		self.data.remove(attributes)
		
		self.tree = self.id3.buildtree(self.data)

	def tearDown(self):
		isFile = os.path.exists('data_set/sample.csv')

		if isFile:
			os.remove('data_set/sample.csv')
			os.rmdir('data_set/')
			isFile = False

		self.assertFalse(isFile)

	def test_instance(self):
		self.assertIsInstance(self.id3, ID3)

	def test_load_sample(self):
		isFile = os.path.exists('data_set/sample.csv')

		self.assertTrue(isFile)

	def test_read_sample(self):
		self.assertTrue(isinstance(self.data, list))

	def test_build_tree(self):
		self.assertIsInstance(self.tree, ID3)

	def test_print_tree(self):
		printtree = self.id3.printtree(self.tree)
		self.assertFalse(printtree)
	
	def test_classify(self):
		classify = self.id3.classify(self.data[9], self.tree)	
		self.assertTrue(isinstance(classify, dict))

	def test_divide_set(self):
		divideset = self.id3.divideset(self.data, 4, 'yes')
		self.assertTrue(isinstance(divideset, tuple))

	def test_unique_counts(self):
		uniquecounts = self.id3.uniquecounts(self.data)
		self.assertTrue(isinstance(uniquecounts, dict))

	def test_entropy(self):
		ent = self.id3.entropy(self.data)
		self.assertTrue(isinstance(ent, int))

if __name__ == '__main__':
    unittest.main()
