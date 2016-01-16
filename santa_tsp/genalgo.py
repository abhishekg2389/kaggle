import multiprocessing
import inspect
import heapq
import random

import numpy as np

class GeneticAlgorithm():
	def __init__(
			self, 
			selection_fn=None,
			mutation_fn=None,
			crossover_fn=None,
			objective_fn=None,
			max_iter=100,
			popu_size=None,
			annealing=False,
			random_seed=0,
			save_popu_frac=0,
			verbose=False):
		
		if selection_fn is None:
			raise Error("Selection Function can't be undefined.")
		else:
			self.selection_fn = selection_fn
		
		if mutation_fn is None:
			raise Error("Mutation Function can't be undefined.")
		else:
			self.mutation_fn = mutation_fn
			
		if crossover_fn is None:
			raise Error("Crossover Function can't be undefined.")
		else:
			self.crossover_fn = crossover_fn
		
		if objective_fn is None:
			raise Error("Objective Function can't be undefined.")
		else:
			self.objective_fn = objective_fn
		
		if not isinstance(save_popu_frac,(int, long, float)):
			raise Error("Population Fraction undefined.")
		else:
			self.save_popu_frac = save_popu_frac
			
		self.max_iter = max_iter
		self.popu_size = popu_size
		self.annealing = annealing
		self.random_seed = random_seed
		random.seed(random_seed)
		self.verbose = verbose
		
	def get_params(self):
		params = {}
		params['selection_fn'] = self.selection_fn
		params['mutation_fn'] = self.mutation_fn
		params['crossover_fn'] = self.crossover_fn
		params['objective_fn'] = self.objective_fn
		params['max_iter'] = self.max_iter
		params['popu_size'] = self.popu_size
		params['save_popu_frac'] = self.save_popu_frac
		params['annealing'] = self.annealing
		params['random_seed'] = self.random_seed
		params['verbose'] = self.verbose
		return params
		
	def fit(self, X):
		if self.popu_size == None:
			self.popu_size = len(X)
		
		n_cpu = multiprocessing.cpu_count()
		
		print "CPU: "+str(n_cpu)
		print " Iter   Best Score"
		print "------  ----------"
		
		counter = 0
		while self.max_iter != counter:
			# --- Evaluation ---
			pool = multiprocessing.Pool(n_cpu)
			objective_objs = pool.map(self.objective_fn, X)
			#objective_objs = map(self.objective_fn, X)
			
			# --- Selection ---
			if isinstance(self.selection_fn, float):
				if self.selection_fn >= 1:
					selected_objs = heapq.nsmallest(self.selection_fn, zip(range(self.popu_size),objective_objs), key=lambda x:x[1])
				else:
					selected_objs = heapq.nsmallest(int(self.popu_size*self.selection_fn), zip(range(self.popu_size),objective_objs), key=lambda x:x[1])
			else:
				selected_objs = self.selection_fn(X, objective_objs)
			selected_objs = [x[0] for x in selected_objs]
			
			# --- Crossover ---
			sel_size = len(selected_objs)
			pool = multiprocessing.Pool(n_cpu)
			crossover_objs = pool.map(self.crossover_fn, [X[random.sample(range(sel_size), 2)] for i in range(self.popu_size-sel_size)])
			# crossover_objs = map(self.crossover_fn, [X[random.sample(range(sel_size), 2)] for i in range(self.popu_size-sel_size)])
			
			# --- Mutation ----
			pool = multiprocessing.Pool(n_cpu)
			pool.map(self.mutation_fn, crossover_objs)
			# map(self.mutation_fn, crossover_objs)
			
			X = np.r_[X[selected_objs], np.array(crossover_objs)]
			counter += 1
			
			print str(counter)+"\t"+str(min(objective_objs))
		
		pool = multiprocessing.Pool(n_cpu)
		objective_objs = pool.map(self.objective_fn, X)
		# objective_objs = map(self.objective_fn, X)
		
		if self.save_popu_frac >= 1:
			top_objs = heapq.nsmallest(int(self.save_popu_frac), zip(range(self.popu_size),objective_objs), key=lambda x:x[1])
			top_objs = [x[0] for x in top_objs]
			self.saved_popu = zip(objective_objs[top_objs], X[top_objs])
			
		elif self.save_popu_frac > 0:
			top_objs = heapq.nsmallest(int(self.popu_size*self.save_popu_frac), zip(range(self.popu_size),objective_objs), key=lambda x:x[1])
			top_objs = [x[0] for x in top_objs]
			self.saved_popu = zip([objective_objs[x] for x in top_objs], X[top_objs])
		
		return self
		
	def best(self):
		return self.saved_population[0][1]

	def best_score(self):
		return self.saved_population[0][0]

	def nbest(self, nbest=1):
		return self.saved_population[:nbest]
