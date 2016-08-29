import random
import math

class PerceptronLinear(object):
	def __init__(self, learning_rate):
		self.threshold = 0
		self.learning_rate = learning_rate
		self.generations = 0
		self.bias = 0
		self.weights = None
		self.errors = []

	def train(self, dataset):
		learned = False
		X = [x[0] for x in dataset]
		Y = [y[1] for y in dataset]

		for xi in X:
			xi.insert(0, 1)

		self.weights = [random.random() for _ in range(len(X[0]))]

		while not learned and self.generations < 10000:
			self.generations += 1
			learned = True
			for i, (xi, yi) in enumerate(zip(X, Y)):
				error = 0
				expected_y = self.perceive(xi)
				if yi != expected_y:
					error = (yi - expected_y)^2
					if error != 0:
						learned = False
						self.updateWeights(error, xi)
				self.errors.append(error)


	def updateWeights(self, error, X):
		for i, (xi, wi) in enumerate(zip(X, self.weights)):
			self.weights[i] += self.learning_rate * error * xi
		self.bias += self.learning_rate * error

	def perceive(self, X):
		if len(X) != len(self.weights):
			X.insert(0, 1)

		result = self.bias + sum([x * w for x, w in zip(X, self.weights)])
		if result > self.threshold:
			return 1
		else:
			return 0