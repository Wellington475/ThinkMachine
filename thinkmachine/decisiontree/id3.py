# -*- coding: utf8 -*-

import os
import csv
from math import log
from random import choice

class ID3:
	def __init__(self, col = -1, value = None, results = None, tb = None, fb=None):
		self.col 	 = col
		self.value 	 = value
		self.results = results
		self.tb 	 = tb
		self.fb 	 = fb

		self.sample = {
			'weather':      ['sunny', 'clouds', 'rain'],
			'temperature': ['hot', 'pleasant', 'cold'],
			'humidity':    ['high', 'normal'],
			'wind':	   	   ['weak', 'strong'],
			'decision':    ['yes', 'no']
		}

	def load_sample(self):
		path = os.path.dirname(os.path.realpath(__file__))+'/../../data_set'
		if not os.path.isdir(path):
			os.makedirs(path)

		with open(path+'/sample.csv', 'w', newline='') as fp:
			file = csv.writer(fp, delimiter=',')
			data = [
					['weather', 'temperature', 'humidity', 'wind', 'decision']
				]

			for i in range(1, 15):
				weather = str(choice(self.sample['weather']))
				temperature = str(choice(self.sample['temperature']))
				humidity = str(choice(self.sample['humidity']))
				wind = str(choice(self.sample['wind']))
				decision = str(choice(self.sample['decision']))

				data.append([weather, temperature, humidity, wind, decision])

			file.writerows(data)

		return('sample.csv')

	def divideset(self, rows, column, value):
		split_set = None
		
		if isinstance(value, int) or isinstance(value, float):
			split_set = lambda row:row[column] >= value
		else:
			split_set = lambda row: row[column] == value

		set1 = [row for row in rows if split_set(row)]
		set2 = [row for row in rows if not split_set(row)]

		return(set1, set2)

	def uniquecounts(self, rows, column = None):
		results = {}
		index = column
		for row in rows:
			if index == None:
				base_column = row[len(row)-1]
			else:
				base_column = row[index]

			if base_column not in results:
				results[base_column] = 0
			
			results[base_column] += 1
		
		return results


	def entropy(self, rows):
		ent = 0.0
		results = self.uniquecounts(rows)
		probs = [float(results[k])/len(rows) for k in results.keys()]

		for p in probs:
			ent -= p * log(p, 2)

		return ent

	def classify(self, observation, tree):
		if tree.results != None:
			i = 0
			for key in tree.results:
				i += 1
				if i == 1:
					return dict({'decision': key})
			i = 0
		else:
			v = observation[tree.col]
			branch = None
			
			if isinstance(v,int) or isinstance(v,float):
				if v >= tree.value:
					branch = tree.tb
				else:
					branch = tree.fb
			else:
				if v == tree.value:
					branch = tree.tb
				else:
					branch = tree.fb

			return self.classify(observation, branch)

	def buildtree(self, rows):
		if len(rows) == 0:
			return decisionnode()
		
		current_score = self.entropy(rows)

		best_gain	  = 0.0
		best_criteria = None
		best_sets	  = None
		base_column  = len(rows[0])-1

		for index in range(0, base_column):
			column_values = {}
			
			for row in rows:
				column_values[row[index]] = 1

			for value in column_values.keys():
				(set1, set2) = self.divideset(rows, index, value)

			prob = float(len(set1)) / len(rows)
			gain = current_score - prob * self.entropy(set1) - (1-prob) * self.entropy(set2)

			if gain > best_gain and len(set1) > 0 and len(set2) > 0:
				best_gain	  = gain
				best_criteria = (index, value)
				best_sets	  = (set1, set2)

		if best_gain>0:
			trueBranch  = self.buildtree(best_sets[0])
			falseBranch = self.buildtree(best_sets[1])
			return ID3(col = best_criteria[0], value = best_criteria[1], tb = trueBranch, fb = falseBranch)
		else:
			return ID3(results = self.uniquecounts(rows))

	def printtree(self, tree, indent=' '):
		if tree.results != None:
			i = 0
			for k in tree.results:
				i += 1
				if i == 1:
					print('Decision: '+str(k))
			i = 0
		else:
			if isinstance(tree.value, int) or isinstance(tree.value, float):
				tree.value = ">="+str(tree.value)
			print('Column['+str(tree.col)+']: '+str(tree.value)+'?')
			print(indent+'True ->', end=" ")
			
			self.printtree(tree.tb, indent+'  ')
			
			print(indent+'False ->', end=" ")
			
			self.printtree(tree.fb, indent+'  ')

	def __repr__(self):
		return("%s - Decision Tree") % (self.__class__.__name__)