import csv
import math
import random
import time
import sys
import multiprocessing
import copy

import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt

from haversine import haversine
from scipy import spatial
from itertools import cycle
from sklearn import cluster
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

# plt.plot(data[:, 2], data[:, 1], ls='None', lw = 0, mfc=col, marker='o', mew = 0, ms=0.5)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()

# ---------------------------------------
# --- Clustering ------------------------
# ---------------------------------------

def dbscan_birch():
	dbscn = cluster.DBSCAN(eps=0.1, min_samples=1, metric=haversine, algorithm='ball_tree', leaf_size=10, p=None)
	labels = dbscn.fit_predict(data[:,[1,2]])
	
	db_clusters = [np.where(labels==i) for i in range(len(np.unique(labels)))]
	
	clusters = []
	for i in range(len(db_clusters)):
		br = cluster.Birch(threshold=0.5, branching_factor=50, n_clusters=None, compute_labels=True, copy=True)
		labels = br.fit_predict(data[db_clusters[i],[1,2]])
		clusters.extend([data[np.where(labels==i)[0], 0].astype(int)-1 for i in range(len(np.unique(labels)))])
	
	return clusters

def longitudinal_split():
	long_srtd_pts = np.argsort(data[:, 1])
	clusters = [[]]
	wt_limit = 900 + random.randint(-50, 50)
	wt = 0
	for i in long_srtd_pts:
		clusters[-1].append(i)
		wt += data[i, 3]
		if wt >= wt_limit:
			clusters.append([])
	
	return clusters

# ---------------------------------------
# --- Clustering Optimization -----------
# ---------------------------------------

def cluster_status(data_pts):
	# 0  : Add more nodes
	# 1  : Remove nodes if possible
	# 1+ : Split into two or more nodes
	
	_cc = np.mean(data_pts[:,[1,2]], axis=0)
	_dt_cc_np = haversine(_cc, [90,0])
	_dt_in_np = haversine(data_pts[0,[1,2]], [90,0])
	_dt_ot_np = haversine(data_pts[-1,[1,2]], [90,0])
	_split = 0
	_step_dist = map(lambda x,y: haversine(x,y), data_pts[:-1][:,[1,2]], data_pts[1:][:,[1,2]])
	_sledge_wrw = 2*10*((_dt_cc_np+_dt_in_np+_dt_ot_np)/3)
	_wrw = 0
	_wts = data_pts[:, 3]
	_dt = 0
	_extra = 0
	_edt = 0
	for j in range(len(data_pts)-1):
		_dt += _step_dist[j]
		_wrw += _wts[j]*_dt + 10*_step_dist[j]
		if _split == 1:
			_extra += _wts[j]*_edt
		if _extra > _sledge_wrw:
			if _split == 1:
				_split = 2
			else:
				print "Not exp...:("
		if _wrw > 1.1*_sledge_wrw:
			_split += 1
			_dt = 0
			_wrw = 0
			if _split == 1:
				_edt = _dt
	
	if _split == 0:
		return _split, _wrw - _sledge_wrw
	else:
		return _split, _wrw

def heuristic_I(routes):
	print "--- Heuristic I ----------------"
	
	_changed = False
	
	_routes = copy.deepcopy([list(x) for x in routes])
	_routes, _wrws = zip(*multiprocessing.Pool().map(greedy_cvrp, [data[_routes[i]] for i in range(len(_routes))]))
	_routes = np.array(_routes)
	_wrws =  np.array(_wrws)
	
	_cluster_centers = np.array([np.mean(data[_routes[i]][:,[1,2]], axis=0) for i in range(len(_routes))])
	_statuses = multiprocessing.Pool().map(cluster_status, [data[_routes[i]] for i in range(len(_routes))])
	_statuses = np.array([list(x) for x in _statuses])
	
	_savings = 0
	_skips = set()
	_splitted = True
	
	while _splitted == True:
		_splitted = False
		
		_vor = spatial.Voronoi(_cluster_centers[:,[1,0]])
		_adjs = [None]*len(_routes)
		for i in range(len(_routes)):
			_adjs[i] = []
		
		for pr in _vor.ridge_points:
			_adjs[pr[0]].append(pr[1])
			_adjs[pr[1]].append(pr[0])
		
		_adjs_down = [None]*len(_routes)
		for i in range(len(_routes)):
			_srtd_adjs, _ = zip(*sorted(zip(_adjs[i], _cluster_centers[_adjs[i]]), key=lambda x:haversine(x[1], (90,0))))
			_srtd_adjs_down = [x for x in _srtd_adjs if _cluster_centers[x,0] < _cluster_centers[i,0] and len(_routes[x]) > 0]
			_adjs_down[i] = _srtd_adjs_down
		
		for i in range(len(_routes)):
			if i in _skips:
				continue
			
			_used_idx = _adjs_down[i]+[i]
			_scr = sum(_statuses[_used_idx, 1])
			_split = len(_used_idx)
			_data_pts = data[np.hstack(np.array(_routes)[_used_idx])]
			_prev_score = sum(_wrws[_used_idx])
			_prev_split = _split
			
			if math.fabs(_scr) < 100000:
				continue
			if _scr > 0:
				_split += 1
			else:
				_split -= 1
			# print "_routes "+str(i)+" "+str(_split)
			
			if _split <= 1:
				_skips.add(i)
				continue
			
			_km = cluster.KMeans(verbose=0, n_jobs=-1, n_clusters=_split, random_state=2389)
			_labels = _km.fit_predict(_data_pts[:,[1,2]])
			_new_cluster_centers = _km.cluster_centers_
			
			_new_routes = [None]*_split
			_new_wrws = [0]*_split
			for j in range(_split):
				_new_routes[j], _new_wrws[j] = greedy_cvrp(_data_pts[np.where(_labels==j)[0]])
			
			if _prev_score < sum(_new_wrws):
				_skips.add(i)
				continue
			if len(np.where(np.array([sum(data[x,3]) for x in _new_routes]) > 1000)[0]) > 0:
				_skips.add(i)
				continue
			if len(np.array(_new_routes).shape) > 1:
				_skips.add(i)
				continue
			
			_savings += _prev_score - sum(_new_wrws)
			print "Net Savings: "+str(_savings)
			
			# print str(len(_routes))+": "+str(_prev_split)+" | "+str(_prev_score)+" --> "+str(_split)+" | "+str(sum(_new_wrws))
			_splitted = True
			_rest_idx = [j for j in range(len(_routes)) if j not in set(_used_idx)]				
			
			_routes = _routes[_rest_idx]
			_routes = np.concatenate((_routes, np.array(_new_routes)), axis = 0)
			_wrws = _wrws[_rest_idx]
			_wrws = np.concatenate((_wrws, np.array(_new_wrws)), axis = 0)
			_cluster_centers = _cluster_centers[_rest_idx]
			_cluster_centers = np.concatenate((_cluster_centers, np.array(_new_cluster_centers)), axis = 0)
			_statuses = _statuses[_rest_idx]
			_statuses = np.concatenate((_statuses, np.array(_new_statuses)), axis = 0)
			
			_new_skips = set()
			for j, x in enumerate(_rest_idx):
				if x in _skips:
					_new_skips.add(j)
			
			_skips = _new_skips
			
			# Sanity Checks
			_lbls = np.zeros((100000,)).astype(int) - 1
			for j, x in enumerate(_routes):
				_lbls[x] = j
			
			if len(np.where(_lbls==-1)[0]) > 0:
				print "Len Error"
				input()
			
			for x in _routes:
				if sum(data[x,3]) > 1000:
					print "Wt Error"
			
			_changed = True
			
			break
	
	print "-------------------------------"
	return _routes, _changed

def heurictic_II(routes):
	print "--- Heuristic II --------------"
	
	_changed = False
	
	_routes = copy.deepcopy([list(x) for x in routes])
	_routes, _wrws = zip(*multiprocessing.Pool().map(greedy_cvrp, [data[_routes[i]] for i in range(len(_routes))]))
	_routes = [list(x) for x in _routes]
	_wrws = np.array(_wrws)
	_cluster_centers = np.array([np.mean(data[_routes[i]][:,[1,2]], axis=0) for i in range(len(_routes))])
	
	_vor = spatial.Voronoi(_cluster_centers[:,[1,0]])
	_adjs = [None]*len(_routes)
	for i in range(len(_routes)):
		_adjs[i] = []
	
	for pr in _vor.ridge_points:
		_adjs[pr[0]].append(pr[1])
		_adjs[pr[1]].append(pr[0])
	
	_adjs_down = [None]*len(_routes)
	for i in range(len(_routes)):
		_srtd_adjs, _ = zip(*sorted(zip(_adjs[i], _cluster_centers[_adjs[i]]), key=lambda x:haversine(x[1], (90,0))))
		_srtd_adjs_down = [x for x in _srtd_adjs if _cluster_centers[x,0] < _cluster_centers[i,0] and len(_routes[x]) > 0]
		_adjs_down[i] = _srtd_adjs_down
	
	_cnts = [len(_routes[i]) for i in range(len(_routes))]
	_statuses = multiprocessing.Pool().map(cluster_status, [data[_routes[i]] for i in range(len(_routes))])
	_extruders = set()
	_labels = np.zeros(len(data), dtype=np.int) - 1
	
	for i, route in enumerate(_routes):
		_labels[route] = i
	
	_npdts = metrics.pairwise_distances(_cluster_centers, np.array([[90,0]]), metric=haversine, n_jobs=-1).flatten()
	_clusters_sorted_idx, _ = zip(*sorted(zip(range(len(_cluster_centers)), _npdts), key=lambda x:x[1]))
	_clusters_sorted_idx = np.array(_clusters_sorted_idx)
	
	_savings = 0
	_counter = 0
	_stat_change = np.zeros((3,3)).astype(int)
	
	print "Initial Score: "+str(sum(_wrws))
	
	for c_idx in _clusters_sorted_idx:
		_counter += 1
		
		if len(_routes[c_idx]) == 0:
			print "Skipped "+str(c_idx)
			continue
		
		if _statuses[c_idx][0] == 1 or _statuses[c_idx][0] > 1:
			_extruders.add(c_idx)
			# continue
		
		_cltrs_idx = _adjs_down[c_idx]+list(_extruders.intersection(_adjs[c_idx]))
		_cltrs_idx = [x for x in _cltrs_idx if _cnts[x] > 0]
		
		if len(_cltrs_idx) == 0:
			continue
		
		_nb_pts = data[np.hstack(np.array(_routes)[_cltrs_idx])]
		_dt_nb_cc = metrics.pairwise_distances(_nb_pts[:,[1,2]], _cluster_centers[[c_idx]], metric=haversine, n_jobs=-1).flatten()
		_srt_args = np.argsort(_dt_nb_cc)
		_wt = np.sum(data[_routes[c_idx], 3])
		
		# --- Prev 
		_prev_score = sum(_wrws[_cltrs_idx+[c_idx]])
		# print "PrevScore: "+str(_prev_score)
		# print "#Cluster: "+str(c_idx)+"\t#pt: "+str(_cnts[c_idx])+"\t    score: "+str(_wrws[c_idx])
		# for x in _adjs_down[c_idx]:
		# 	print "#Cluster: "+str(x)+"\t#pt: "+str(_cnts[x])+"\t    score: "+str(_wrws[x])
		
		print str(_counter)+"\t"+str(_savings)+"\t"+str(len(_nb_pts))+"\t"+str(len(_cltrs_idx))
		
		_altrd = set()
		
		for i in _srt_args:
			if _wt+_nb_pts[i,3] > 1000:
				continue
			
			_it = int(_nb_pts[i,0])-1
			_fm = _labels[_it]
			_to = c_idx
			_r1 = copy.deepcopy(_routes[_fm])
			_r2 = copy.deepcopy(_routes[_to])
			_r1.remove(_it)
			_r2.append(_it)
			
			__w1 = _wrws[_fm]
			__w2 = _wrws[_to]
			
			if len(_r1) == 0:
				_r1, _w1 = [], 0
			else:
				_r1, _w1 = greedy_cvrp(data[_r1])
			_r2, _w2 = greedy_cvrp(data[_r2])
			
			if _w1+_w2 < __w1+__w2:
				_wt += _nb_pts[i,3]
				_routes[_fm] = list(_r1)
				_routes[_to] = list(_r2)
				_wrws[_fm] = _w1
				_wrws[_to] = _w2
				_altrd.add(_fm)
				_altrd.add(_to)
				_s2 = cluster_status(data[_r2])
				_labels[_it] = _to
				_changed = True
				if _s2[0] >= 1:
					_extruders.add(c_idx)
					# break
		
		# --- Route lengths change
		_cnts[c_idx] = len(_routes[c_idx])
		for x in _altrd:
			_cnts[x] = len(_routes[x])
		
		# --- Clusters Centers change
		_cluster_centers[c_idx] = np.mean(data[_routes[c_idx]][:, [1,2]], axis=0)
		for x in _altrd:
			if _cnts[x] > 0:
				_cluster_centers[x] = np.mean(data[_routes[x]][:, [1,2]], axis=0)
		
		# --- Change Adj DOWN
		for x in _altrd:
			if _cnts[x] == 0:
				if x in _adjs_down[c_idx]:
					_adjs_down[c_idx].remove(x)
				for y in _adjs[x]:
					if x in _adjs_down[y]:
						_adjs_down[y].remove(x)
		
		# --- Extruder
		for x in _altrd:
			if _cnts[x] == 0:
				if x in _extruders:
					_extruders.remove(x)
		
		# --- Change Statuses
		for x in _altrd:
			if_cnts[x] == 0:
				continue
			_new_stat = cluster_status(data[_routes[x]])
			if _new_stat[0] != _statuses[x][0]:
				_fm_stt = _statuses[x][0]
				_to_stt = _new_stat[0]
				if _fm_stt > 2:
					_fm_stt = 2
				if _to_stt > 2:
					_to_stt = 2
				_stat_change[_fm_stt][_to_stt] += 1
			_statuses[x] = _new_stat
		
		# --- Final
		_final_score = sum(_wrws[_cltrs_idx+[c_idx]])
		# print "FinalScore: "+str(_final_score)
		# print "#Cluster: "+str(c_idx)+"\t#pt: "+str(_cnts[c_idx])+"\t    score: "+str(_wrws[c_idx])
		# for x in _adjs_down[c_idx]:
		# 	print "#Cluster: "+str(x)+"\t#pt: "+str(_cnts[x])+"\t    score: "+str(_wrws[x])
		
		_savings += _prev_score - _final_score
		
		if _final_score > _prev_score:
			print "Error...:("+str(c_idx)
	
	print "-------------------------------"
	return [list(x) for x in _routes if len(x) > 0], _changed

def heuristic_III(routes):
	print "--- Heuristic III -------------"
	
	_chnaged = False
	
	_routes = copy.deepcopy([list(x) for x in routes])
	_routes, _wrws = zip(*multiprocessing.Pool().map(greedy_cvrp, [data[_routes[i]] for i in range(len(_routes))]))
	_cluster_centers = np.array([np.mean(data[_routes[i]][:,[1,2]], axis=0) for i in range(len(_routes))])
	_routes = [list(x) for x in _routes]
	_wrws = list(_wrws)
	_vor = spatial.Voronoi(_cluster_centers[:,[1,0]])
	_savings = 0
	
	for _, pr in enumerate(_vor.ridge_points):
		# print _
		_altrd = set()
		_hvr1 = np.array(metrics.pairwise_distances(data[_routes[pr[0]]][:,[1,2]], np.array([_cluster_centers[pr[0]]]), metric=haversine, n_jobs=-1).flatten())
		_hvr2 = np.array(metrics.pairwise_distances(data[_routes[pr[1]]][:,[1,2]], np.array([_cluster_centers[pr[1]]]), metric=haversine, n_jobs=-1).flatten())
		
		_tops1 = list(np.array(_routes[pr[0]])[np.argsort(_hvr1)][::-1])
		_tops2 = list(np.array(_routes[pr[1]])[np.argsort(_hvr2)][::-1])
		
		_r1 = copy.deepcopy(_routes[pr[0]])
		_r2 = copy.deepcopy(_routes[pr[1]])
		
		_upto = 10
		_up = True
		i = 0
		j = 0
		
		while True:
			
			if i+j == _upto:
				break
			
			if _up:
				if i == 0:
					j += 1
					_up = False
				else:
					i -= 1
					j += 1
			else:
				if j == 0:
					i += 1
					_up = True
				else:
					i += 1
					j -= 1
			
			if i >= len(_tops1) or j >= len(_tops2):
				continue
			
			if _tops1[i] not in _r1 or _tops2[j] not in _r2:
				continue
			
			_n1 = _tops1[i]
			_n2 = _tops2[j]
			
			_r1.remove(_n1)
			_r2.append(_n1)
			
			_r2.remove(_n2)
			_r1.append(_n2)
			
			if sum(data[_r1, 3]) > 1000 or sum(data[_r2, 3]) > 1000:
				_r1.append(_n1)
				_r2.remove(_n1)
				
				_r2.append(_n2)
				_r1.remove(_n2)
				continue
			
			__w1 = _wrws[pr[0]]
			__w2 = _wrws[pr[1]]
			
			_r1, _w1 = greedy_cvrp(data[_r1])
			_r2, _w2 = greedy_cvrp(data[_r2])
			
			_r1 = list(_r1)
			_r2 = list(_r2)
			
			if _w1+_w2 < __w1+__w2:
				_savings += (__w1+__w2) - (_w1+_w2)
				_routes[pr[0]] = copy.deepcopy(_r1)
				_routes[pr[1]] = copy.deepcopy(_r2)
				_wrws[pr[0]] = _w1
				_wrws[pr[1]] = _w2
				_altrd.add(pr[0])
				_altrd.add(pr[1])
				_cluster_centers[pr[0]] = np.mean(data[_routes[pr[0]]][:, [1,2]], axis=0)
				_cluster_centers[pr[1]] = np.mean(data[_routes[pr[1]]][:, [1,2]], axis=0)
				
				_tops1.remove(_n1)
				_tops2.append(_n1)
				_tops2.remove(_n2)
				_tops1.append(_n2)
				
				_changed = True
				
				print "Net Savings: "+str(_savings)
			else:
				_r1.append(_n1)
				_r2.remove(_n1)
				
				_r2.append(_n2)
				_r1.remove(_n2)
				# print (__w1+__w2) - (_w1+_w2)
	
	print "-------------------------------"
	return _routes, _changed

# ---------------------------------------
# --- Routes Optimization ---------------
# ---------------------------------------

# --- Greedy Approach -------------------
def greedy_cvrp(data_pts):
	t0 = time.time()
	# print "Cluster... "+str(i+1)+" out of "+str(_n_cltrs)
	_wt = data_pts[:,3]
	_hvr = metrics.pairwise_distances(data_pts[:,[1,2]], metric=haversine, n_jobs=1)
	_wt_hvr = _wt*_hvr
	_mx = [np.max(_wt_hvr)+1]*len(data_pts)
	_bst_route = []
	_bst_wrw = sys.maxint	
	for j in range(len(data_pts)):
		_route = []
		_route.append(j)
		_mask = np.zeros(len(data_pts))
		_mask[j] = 1
		_agmn = j
		
		_temp_wt = np.sum(data_pts[:,3])
		_strt_dt = haversine((90,0),data_pts[j,[1,2]])
		_wrw = (_temp_wt+10)*_strt_dt
					
		while sum(_mask) != len(data_pts):
			_agmn = np.argmin(_hvr[_agmn,:]+_mx*_mask)
			_route.append(_agmn)
			_mask[_agmn] = 1
			_temp_wt -= data_pts[_route[-2],3]
			_wrw += (_temp_wt+10)*_hvr[_route[-2], _route[-1]]				
			if _wrw > _bst_wrw:
				break
		
		_end_dt = haversine(data_pts[_route[-1],[1,2]], (90,0))
		_wrw += 10*_end_dt
		
		if _wrw < _bst_wrw:
			_bst_route = data_pts[_route,0].astype(int)-1
			_bst_wrw = _wrw
	
	# print "Time Taken SubOpt: "+str(time.time() - t0)
	return _bst_route, _bst_wrw

# --- Exact CVRP with Branch & Bound ----
def exact_cvrp(data_pts):
	greedy_cvrp, subopt_wrw = greedy_cvrp(data_pts)
	# print subopt_wrw
	
	_exact_wrw = 0
	_mask = np.zeros(len(data_pts), dtype=bool)
	_sel = []
	_sel_ = range(len(data_pts))
	_np_hvr = metrics.pairwise_distances(data_pts[:, [1,2]], np.array([[90,0]]), metric=haversine, n_jobs=1).flatten()
	_hvr = metrics.pairwise_distances(data_pts[:,[1,2]], metric=haversine, n_jobs=1)
	_iwts = data_pts[:,3]
	_ids = data_pts[:,0]
	_np_hvr_argsort = np.argsort(_np_hvr)
	_hvr_argsort = np.argsort(_hvr, axis=1)
	_iwts_argsort = np.argsort(_iwts)
	
	_best_route = []
	_best_wrw = subopt_wrw + 1
	
	_wts = [None]*(len(data_pts)+1)
	_wrws = [None]*(len(data_pts)+1)
	_wts[0] = 10+sum(_iwts)
	_wrws[0] = 0
	_back = False
	_counter = 0
	
	while True:
		_counter += 1
		# if _counter % 100 == 0:
		# 	print _sel
		
		_res = exact_cvrp_iter(_sel, _sel_, _back)
		
		if not _res:
			if _back == True:
				break
			
			if _back == False:
				# print _sel
				# print _sel_
				_fn_wrw = haversine(data_pts[_sel[-1], [1,2]], (90,0))*10
				if _wrws[len(_sel)]+_fn_wrw < _best_wrw:
					_best_wrw = _wrws[len(_sel)]+_fn_wrw
					_best_route = list(_sel)
					print _best_wrw
					# print _best_route
					
				_back = True
				continue
		else:
			if _back == True:
				_back = False
		
		if len(_sel) == 1:
			_dt = haversine((90,0), data_pts[_sel[0], [1,2]])
		elif len(_sel) > 1:
			_dt = _hvr[_sel[-2], _sel[-1]]
		else:
			_dt = None
		
		_wrws[len(_sel)] = _wrws[len(_sel)-1] + _wts[len(_sel)-1]*_dt
		_wts[len(_sel)] = _wts[len(_sel)-1] - data_pts[_sel[-1], 3]
		
		if _wrws[len(_sel)]+exact_cvrp_low_bound(_sel, _sel_, _np_hvr, _np_hvr_argsort, _hvr, _hvr_argsort, _iwts, _iwts_argsort, _ids) > _best_wrw:
			_back = True
		else:
			_back = False
	
	# print _counter, subopt_wrw - _best_wrw
	
	_best_route = data_pts[_best_route, 0].astype(int) - 1
	return _best_route, _best_wrw

def exact_cvrp_iter(sel, _sel, back):
	if back:
		while True:
			if not sel:
				return False
			pop = sel.pop()
			_sel.append(pop)
			nxt = [x for x in _sel if x > pop]
			if not nxt:
				continue
			else:
				sel.append(min(nxt))
				_sel.remove(min(nxt))
				break
	else:
		if not _sel:
			return False
		sel.append(min(_sel))
		_sel.remove(min(_sel))
	
	return True

def exact_cvrp_low_bound(sel, _sel, np_hvr, np_hvr_argsort, hvr, hvr_argsort, iwts, iwts_argsort, ids):
	if not _sel:
		return haversine(data[ids[sel[-1]]-1, [1,2]], (90,0))*10
	
	_set_sel = set(_sel)
	
	_min2 = [x for x in np_hvr_argsort if x in _set_sel]
	if not sel:
		if len(_min2) == 1:
			_in = np_hvr[_min2[0]]
			_out = np_hvr[_min2[0]]
		else:
			_in = np_hvr[_min2[0]]
			_out = np_hvr[_min2[1]]
	else:
		_in = hvr[sel[-1], [x for x in hvr_argsort[sel[-1]] if x in _set_sel][0]]
		_out = np_hvr[_min2[0]]
	
	if len(_sel) == 1:
		_bound = _in*(iwts[_sel[0]]+10)
		_bound += _out*10
		return _bound
	
	if len(_sel) == 2:		
		if iwts[_sel[0]] > iwts[_sel[1]]:
			mx_id = 0
			mn_id = 1
		else:
			mx_id = 1
			mn_id = 0
		_bound = _in*(iwts[_sel[0]]+iwts[_sel[1]]+10)
		_bound += hvr[_sel[mx_id], _sel[mn_id]]*(iwts[_sel[mn_id]]+10)
		_bound += _out*10
		return _bound
	
	_sel_wts_idx = np.array([x for x in iwts_argsort if x in _set_sel])[::-1]
	_cum_iwts = np.cumsum(iwts[_sel_wts_idx[::-1]])[::-1] + 10
	
	_sel_hvr_idx = [[y for y in hvr_argsort[x] if y in _set_sel] for x in _sel_wts_idx]
	_lowers2 = np.array([hvr[_sel_wts_idx[i], x[1:3]] for i, x in enumerate(_sel_hvr_idx)])
	_lower2 = _lowers2[:,1]
	_lower1 = _lowers2[:,0]
	
	_bound = _in*_cum_iwts[0]
	_bound += (_lower1[0]*_cum_iwts[1]+np.dot(_lower1[1:-1],_cum_iwts[1:-1])+np.dot(_lower2[1:-1],_cum_iwts[2:])+_lower1[-1]*_cum_iwts[-1])/2
	_bound += _out*10
	
	return _bound

# --- Genetic Algo ----------------------
def mut_fn(route):
	mut_perc = 0.6
	
	if mut_perc < 1:
		mut_perc = int((mut_perc*len(route))/2)

	for _ in range(mut_perc):
		i, j = random.sample(range(len(route)),2)
		route[i], route[j] = route[j], route[i]
	
def crs_fn(parent_routes):
	i, j = random.sample(range(len(parent_routes[0])),2)
	sp0 = set(parent_routes[0][i:j])
	child = [x for x in parent_routes[1] if x not in sp0]
	child.extend(parent_routes[0][i:j])
		
	return child

def gen_algo(data_pts):
	_greedy_route, _greedy_wrw = greedy_cvrp(data_pts)
	_route_len = len(data_pts)
	_random_seed_data = np.array([random.sample(range(_route_len), _route_len) for i in range(_route_len)])
	
	_ga = GeneticAlgorithm(selection_fn=0.2, mutation_fn=mut_fn, crossover_fn=crs_fn, objective_fn=wrw, max_iter=10,
			popu_size=_route_len, annealing=False , save_popu_frac=0, verbose=True)
	# print _ga.get_params()
	_t0 = time.time()
	_ga.fit(np.r_[_greedy_route, _random_seed_data])
	print "Time Taken: "+str(time.time() - _t0)
	
	return _ga.best(), wrw(_ga.best())

# ---------------------------------------
# --- Evaluation ------------------------
# ---------------------------------------

def wrw(route):
	_pts = data[route]
	_wt = sum(_pts[:,3])
	_strt_dt = haversine((90,0), _pts[0,[1,2]])
	_wrw = (_wt+10)*_strt_dt
	
	for i in range(1,len(route)):
		_wt -= _pts[i-1,3]
		_wrw += (_wt+10)*haversine(_pts[i-1, [1,2]], _pts[i, [1,2]])
	
	_end_dt = haversine(_pts[-1, [1,2]], (90,0))
	_wrw += 10*_end_dt
	
	return _wrw

# ---------------------------------------
# --- Plotting --------------------------
# ---------------------------------------

def clusters_plot(routes):
	_cc = np.array([np.mean(data[routes[i]][:,[1,2]], axis=0) for i in range(len(routes))])
	_col = cycle(colors.cnames.keys())
	_vor = spatial.Voronoi(_cc[:,[1,0]])
	spatial.voronoi_plot_2d(_vor)
	plt.show()
	for x, col in zip(routes, _col):
		_ = plt.plot(data[x, 2], data[x, 1], ls='None', lw = 0, mfc=col, marker='o', mew = 0, ms=0.5)
	
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	# plt.savefig('1.png')
	plt.show()

def voronoi_plot(routes):
	plt.xlim((-200, 200))
	plt.ylim((-100, 100))
	plt.hold(True)
	
	plt.plot(_cc[:,1], _cc[:,0], 'bo', ms=0.5)
	plt.plot(_vor.vertices[:,0], _vor.vertices[:,1], 'ko', ms=1)
	
	for vpair in _vor.ridge_vertices:	
		if vpair[0] >= 0 and vpair[1] >= 0:
			_v0 = _vor.vertices[vpair[0]]
			_v1 = _vor.vertices[vpair[1]]
			_ = plt.plot([_v0[0], _v1[0]], [_v0[1], _v1[1]], 'k', linewidth=0.1)
	
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	# plt.savefig('1.svg')
	plt.show()

# ---------------------------------------
# --- Submission ------------------------
# ---------------------------------------

def submit(routes):
	with open('Submission.'+str(1)+'.csv', 'wb') as out:
		writer = csv.writer(out)
		writer.writerow(['GiftId','TripId'])
		for i, x in enumerate(routes):
			for y in x:
				writer.writerow([y+1, i+1])

# ---------------------------------------
# --- Main ------------------------------
# ---------------------------------------

n_runs = 10
best_wrws = sys.maxint
best_routes = []
for i in range(n_runs):
	random.seed(i)
	
	# routes = dbscan_birch()
	routes = longitudinal_split()
	
	changed = True
	while changed:
		routes, chngI = heuristic_I(routes)
		routes, chngII = heuristic_II(routes)
		routes, chngIII = heuristic_III(routes)
		
		changed = chngI or chngII or chngIII
	
	wrws = multiprocessing.Pool().map(wrw, [data[routes[i]] for i in range(len(routes))])
	if wrws < best_wrw:
		best_wrws = wrws
		best_routes = routes
		
	# clusters_plot(routes)
	# voronoi_plot(routes)
	
submit(best_routes)
