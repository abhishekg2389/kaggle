import csv
import math
import cPickle as pickle
import random
from haversine import haversine

from genalgo import GeneticAlgorithm

# ---------------------------------------
# --- Data Processing -------------------
# ---------------------------------------
data = []
with open("gifts.csv", 'rb') as inp:
	csvReader = csv.reader(inp)
	for row in csvReader:		
		data.append(row)

data = data[1:]
for i in range(len(data)):
	data[i][0] = float(data[i][0])
	data[i][1] = float(data[i][1])
	data[i][2] = float(data[i][2])
	data[i][3] = float(data[i][3])
data = np.array(data)

# plt.hist(data[:,3], bins=50, histtype='bar')
# plt.show()

# ---------------------------------------
# --- Genetic Algorithm -----------------
# ---------------------------------------

def mut_fn(route):
	mut_perc = 4
	
	if mut_perc < 1:
		mut_perc = (mut_perc*len(route))/2

	for _ in range(mut_perc):
		i, j = random.sample(range(len(route)),2)
		route[i], route[j] = route[j], route[i]
	
def crs_fn(p1, p2):
	i, j = random.sample(range(len(p1)),2)
	child = [x for x in p2 if x not in p1[i:j]]
	for x in range(i,j):
		child.insert(x,p1[i])
		
	return child
		
def obj_fn(route):
	dist = 0
	for i in range(1,len(route)):
		dist += haversine(data[i-1,[1,2]], data[i,[1,2]])
	return dist
	
# --- Seed Data -------------- First 40
seed_data = np.array([random.sample(range(40),40) for i in range(1000)])

ga = GeneticAlgorithm(selection_fn=0.8,mutation_fn=mut_fn,crossover_fn=crs_fn,objective_fn=obj_fn,max_iter=100,popu_size=None,annealing=False,
			random_seed=0,save_popu_frac=0,verbose=True)

ga.fit(seed_data)

# ---------------------------------------
# --- Submission ------------------------
# ---------------------------------------
with open('Submission.'+str(1)+'.csv', 'wb') as out:
	writer = csv.writer(out)
	writer.writerow(['GiftId','TripId'])
	writer.writerows(sorted(result.items()))
