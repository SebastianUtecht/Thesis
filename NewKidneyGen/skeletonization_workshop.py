import numpy as np
import random
from scipy.spatial import distance
from vispy import scene, color
from vispy.scene import visuals
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as spatial
# import vg
import time
from scipy.sparse.csgraph import connected_components
import plotly.graph_objects as go
import os


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


np.random.seed(0)

# def load_tree(path, scene=-1, cell_type=True):
# 	data = np.load(path, allow_pickle=True)
# 	x, p = data[:2]
# 	if len(data) > 2:
# 		cell_type = data[3]
# 		tree_scene_x, tree_scene_p, tree_scene_cell_type = x[scene], p[scene], cell_type[scene]
# 		mask = tree_scene_cell_type != 2
# 		x = tree_scene_x[mask]
# 		p = tree_scene_p[mask]
# 	return x, p

def load_tree(path, frame=-1):
	data = np.load(path, allow_pickle=True)
	p_mask_lst, x_lst, p_lst, q_lst = data
	p_mask, x, p, q = p_mask_lst[frame], x_lst[frame], p_lst[frame], q_lst[frame]
	p_mask[p_mask == 1] = 0	
	p_mask[p_mask == 2] = 1

	x_polar = x[p_mask == 1]
	p_polar = p[p_mask == 1]

	return x_polar, p_polar


# Function to find the closest point on the line for multiple indices



def closest_point_on_lines(x, p, threshold, indices):

	results = []
	k=0
	p = p / np.linalg.norm(p, axis=1)[:, None]
	for index in indices:
		
		# Selected point and polarity
		x0 = x[index]
		p0 = p[index]
		k+=1

		# find all the points within the threshold distance of the line spanned by the selected point x0 and the direction p0
		# Distance from the points in x to the line spanned by x0 and p0
		cross_products = np.cross(x - x0, p0)
		distances = np.linalg.norm(cross_products, axis=1)
		# set any 0 distance equal to a very large number
		distances[distances == 0] = 1e9

		

		# Find the points within the threshold distance of the line spanned by x0 and p0
		indices_close_to_line = np.where(distances < threshold)[0]
		points_close_to_line = x[indices_close_to_line]


		for i in range(1,20):
			x_new = x0 - p0*i
			# check if any of the points in points_close_to_line are within a distance of threshold from x_new
			close_to_line_to_x_new = points_close_to_line - x_new
			distance_to_x_new = np.linalg.norm(close_to_line_to_x_new, axis=1)

			valid_indices = distance_to_x_new < 2
			if np.sum(valid_indices) == 0:
				continue
			indices_close_to_line = indices_close_to_line[valid_indices]
			# Sort the filtered points and distances
			sorted_indices = np.argsort(distance_to_x_new[valid_indices])
			indices_close_to_line = indices_close_to_line[sorted_indices]
			points_close_to_line_new = x[indices_close_to_line]
			results.append((x0, p0, indices_close_to_line, points_close_to_line, points_close_to_line_new[0]))
			break


	midpoints = np.empty((0, 3))

	for (x0, p0, closest_point, closest_index, points_close_to_line_new) in results:
		if closest_point is not None:

			# Midpoint
			midpoint = (x0 + points_close_to_line_new) / 2
			midpoints = np.vstack((midpoints, midpoint))
	return results, midpoints



def thin_line(points, point_cloud_thickness=0.53, iterations=1,sample_points=0):
	
	# Iterative thinning of points
	for i in range(iterations):
		# print(f'Iteration {i+1} out of {iterations}')
		if sample_points != 0:
			points = points[:sample_points]
		
		# Sort points into KDTree for nearest neighbors computation later
		point_tree = spatial.cKDTree(points)

		# Empty array for transformed points
		new_points = []
		# Empty array for regression lines corresponding ^^ points
		regression_lines = []
		for point in point_tree.data:
			# Get list of points within specified radius {point_cloud_thickness}
			start_time = time.perf_counter()
			points_in_radius = point_tree.data[point_tree.query_ball_point(point, point_cloud_thickness)]

			# Get mean of points within radius
			start_time = time.perf_counter()
			data_mean = points_in_radius.mean(axis=0)

			# Calulate 3D regression line/principal component in point form with 2 coordinates
			uu, dd, vv = np.linalg.svd(points_in_radius - data_mean)
			linepts = vv[0] * np.mgrid[-1:1:2j][:, np.newaxis]
			linepts += data_mean
			regression_lines.append(list(linepts))

			# Project original point onto 3D regression line
			ap = point - linepts[0]
			ab = linepts[1] - linepts[0]
			point_moved = linepts[0] + np.dot(ap,ab) / np.dot(ab,ab) * ab

			new_points.append(list(point_moved))
		points = np.array(new_points)
	# delete thinned points more than 1.5 units away from other points
	distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
	neighbor_counts = np.sum((distances <= 1) & (distances > 0), axis=1)
	points = points[neighbor_counts > 0]
	# delete the same indices from the regression lines
	regression_lines = [regression_lines[i] for i in range(len(regression_lines)) if neighbor_counts[i] > 0]




	return points, regression_lines


# Sorts points outputed from thin_points()
def sort_points(points, regression_lines, start_index = 522, sorted_point_distance=0.2):
	sort_points_time = time.perf_counter()
	# Index of point to be sorted
	index = start_index

	# sorted points array for left and right of intial point to be sorted
	sort_points_left = [points[index]]
	sort_points_right = []

	# Regression line of previously sorted point
	regression_line_prev = regression_lines[index][1] - regression_lines[index][0]

	# Sort points into KDTree for nearest neighbors computation later
	point_tree = spatial.cKDTree(points)


	# Iterative add points sequentially to the sort_points_left array
	while 1:
		# Calulate regression line vector; makes sure line vector is similar direction as previous regression line
		v = regression_lines[index][1] - regression_lines[index][0]
		if np.dot(regression_line_prev, v ) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v))  < 0:
			v = regression_lines[index][0] - regression_lines[index][1]
		regression_line_prev = v

		# Find point {distR_point} on regression line distance {sorted_point_distance} from original point 
		distR_point = points[index] + ((v / np.linalg.norm(v)) * sorted_point_distance)

		# Search nearest neighbors of distR_point within radius {sorted_point_distance / 3}
		points_in_radius = point_tree.data[point_tree.query_ball_point(distR_point, sorted_point_distance / 3)]
		if len(points_in_radius) < 1:
			break
		# print(f'Number of points in radius: {len(points_in_radius)}')
		# Neighbor of distR_point with smallest angle to regression line vector is selected as next point in order
		# CAN BE OPTIMIZED
		# 
		nearest_point = points_in_radius[0]
		distR_point_vector = distR_point - points[index]
		nearest_point_vector = nearest_point - points[index]
		for x in points_in_radius: 
			x_vector = x - points[index]
			if angle_between(distR_point_vector, x_vector) < angle_between(distR_point_vector, nearest_point_vector):
				nearest_point_vector = nearest_point - points[index]
				nearest_point = x
				
		index = np.where(points == nearest_point)[0][0]

		# Add nearest point to 'sort_points_left' array
		sort_points_left.append(nearest_point)
	# Do it again but in the other direction of initial starting point 
	index = start_index
	regression_line_prev = regression_lines[index][1] - regression_lines[index][0]
	while 1:
		# Calulate regression line vector; makes sure line vector is similar direction as previous regression line
		v = regression_lines[index][1] - regression_lines[index][0]
		if np.dot(regression_line_prev, v ) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v))  < 0:
			v = regression_lines[index][0] - regression_lines[index][1]
		regression_line_prev = v

		# Find point {distR_point} on regression line distance {sorted_point_distance} from original point 
		# Now vector is substracted from the point to go in other direction
		distR_point = points[index] - ((v / np.linalg.norm(v)) * sorted_point_distance)

		# Search nearest neighbors of distR_point within radius {sorted_point_distance / 3}
		points_in_radius = point_tree.data[point_tree.query_ball_point(distR_point, sorted_point_distance / 3)]
		if len(points_in_radius) < 1:
			break

		# Neighbor of distR_point with smallest angle to regression line vector is selected as next point in order
		# CAN BE OPTIMIZED
		# 
		nearest_point = points_in_radius[0]
		distR_point_vector = distR_point - points[index]
		nearest_point_vector = nearest_point - points[index]
		for x in points_in_radius: 
			x_vector = x - points[index]
			if angle_between(distR_point_vector, x_vector) < angle_between(distR_point_vector, nearest_point_vector):
				nearest_point_vector = nearest_point - points[index]
				nearest_point = x
		index = np.where(points == nearest_point)[0][0]

		# Add next point to 'sort_points_right' array
		sort_points_right.append(nearest_point)

	if len(sort_points_left) == 0:
		return np.array(sort_points_right)
	if len(sort_points_right) == 0:
		return np.array(sort_points_left)
	#if len(sort_points_left) == 0 and len(sort_points_right) == 0:
	#	break

	# calculate distance from sort_points_left[-1] to sort_points_right[0] and the distance from sort_points_left[0] to sort_points_right[0]
	# print(f'length of left: {len(sort_points_left)} and lenght of right: {len(sort_points_right)}')
	dist_left_to_right = np.linalg.norm(sort_points_left[-1] - sort_points_right[0])
	dist_start_to_right = np.linalg.norm(sort_points_left[0] - sort_points_right[0])

	
	# if the distance from sort_points_left[-1] to sort_points_right[0] is smaller than the distance from sort_points_left[0] to sort_points_right[0]
	# combine the two arrays in the order of sort_points_left + sort_points_right
	# else combine the two arrays in the order of sort_points_right + sort_points_left
	if dist_left_to_right < dist_start_to_right:
		sorted_points_combined = sort_points_left + sort_points_right
	else:
		sorted_points_combined = sort_points_left[::-1] + sort_points_right

	# print("--- %s seconds to sort points ---" % (time.perf_counter() - sort_points_time))
	return np.array(sorted_points_combined)

# make connected lines in every segment
def sort_points_in_segments(thinned_points, regression_lines, sorted_points_distance=2):
	
	allowed_indices = list(range(len(thinned_points)))

	# collect the sorted points in a list
	sorted_points_collected = []
	k = 0
	while True:
		sorted_points = sort_points(thinned_points[allowed_indices], [regression_lines[i] for i in allowed_indices], start_index = np.random.choice(len(thinned_points[allowed_indices])), sorted_point_distance=sorted_points_distance)
		if len(sorted_points) > 1: # discard lines with only one point
			k+=1
			# print('line_segment_count',k)
			sorted_points_collected.append(sorted_points)

		# delete all the points in thinned_points that are within sorted_points_distance of the sorted_points
		distances = np.linalg.norm(thinned_points[:, None] - sorted_points, axis=2)
		indices = np.where(np.min(distances, axis=1) < sorted_points_distance)[0]
		allowed_indices = [i for i in allowed_indices if i not in indices]
		# print(len(allowed_indices))
		if len(allowed_indices) < 1: # when no points are left we end the loop and return the line collections
			break
	return sorted_points_collected


def create_adjacency_matrix(sorted_points_collected):
	'''
	Takes in the sorted points in a list of arrays format and returns an adjacency matrix where each point is connected to the next point in the array. 
	Endpoints are connected only to the next or previous point in the array. 
	Also return the points array that correspond to the adjacency matrix and the endpoints of the segments
	'''
	# Flatten the list of arrays into a single array of points
	points = np.vstack(sorted_points_collected)
	adjacency_list = {}
	endpoints = []  # To store the indices of the endpoints

	current_index = 0
	for arr in sorted_points_collected:
		num_points = len(arr)

		# Add connections for consecutive points in the current array
		for i in range(num_points - 1):
			if current_index + i not in adjacency_list:
				adjacency_list[current_index + i] = []
			if current_index + i + 1 not in adjacency_list:
				adjacency_list[current_index + i + 1] = []

			# Connect point i with point i+1
			adjacency_list[current_index + i].append(current_index + i + 1)
			adjacency_list[current_index + i + 1].append(current_index + i)

		# Save the endpoints (first and last points of this segment)
		endpoints.append(current_index)  # First point
		endpoints.append(current_index + num_points - 1)  # Last point

		# Update current_index for the next array
		current_index += num_points

	# build the adjacency matrix
	adjacency_matrix = np.zeros((len(points), len(points)))
	for key, neighbors in adjacency_list.items():
		for neighbor in neighbors:
			adjacency_matrix[key, neighbor] = 1
	
	return adjacency_matrix, endpoints, points


def calculate_midpoints_of_three_endpoints(endpoints, adjacency_matrix, points, distance_threshold=10):
	triplets = combinations(endpoints, 3)
	valid_triplets = []
	for triplet in triplets:
		# Make a conditional that checks if the any of the triplets are connected through other points


		coords = points[np.array(triplet)]
		midpoint = np.mean(coords, axis=0)
		distances_to_midpoint = np.linalg.norm(coords - midpoint, axis=1)
		
		# Check if all points in the triplet are sufficiently close to the midpoint
		if np.all(distances_to_midpoint < distance_threshold):
			# Check if the triplet is connected through other points in the graph within the distance threshold
			n_components, labels = connected_components(adjacency_matrix, directed=False)
			i, j, k = triplet
			if labels[i] == labels[j] or labels[j] == labels[k] or labels[i] == labels[k]:
				continue
			valid_triplets.append((triplet, midpoint))
	return valid_triplets
	
def connect_triplet_endpoints(adjacency_matrix, endpoints, points, radius):
	number_unpaired_endpoints = len(endpoints)
	# keep only the endpoints that are part of a subgraph with at least 3 nodes (i.e. not the endpoints of a single line)
	# This cannot be done with adjacency matrix, since the sum of any row is always 1 for an endpoint


	
	number_of_triplet_comb =  (number_unpaired_endpoints * (number_unpaired_endpoints - 1) * (number_unpaired_endpoints - 2) / 6)
	used_index = []
	if number_of_triplet_comb < 1000000:
		valid_triplets = calculate_midpoints_of_three_endpoints(endpoints, adjacency_matrix, points, distance_threshold=radius)
		# print(f'valid triplet: {valid_triplets}')
		for triplet, midpoint in valid_triplets:
			if triplet[0] in used_index or triplet[1] in used_index or triplet[2] in used_index:
				# print('used index')
				continue
			# check if the vector from each of the points to the midpoint is parallel to the vector going from the points connected to the endpoints to the endpoints
			parallel = True
			for i in range(3):
				endpoint = triplet[i]

				# Find the point connected to the current endpoint
				connected_indices = np.where(adjacency_matrix[endpoint] == 1)[0]
				if len(connected_indices) == 0:
					# print(f"No connected point found for endpoint {endpoint}. Skipping parallelism check.")
					parallel = False
					break

				# Assume the first connected point is the one to compare
				connected_point = connected_indices[0]

				# Vector from the connected point to the endpoint
				vector_to_endpoint = points[endpoint] - points[connected_point]
				vector_to_endpoint /= np.linalg.norm(vector_to_endpoint)  # Normalize

				# Vector from the endpoint to the midpoint
				vector_to_midpoint = midpoint - points[endpoint]
				vector_to_midpoint /= np.linalg.norm(vector_to_midpoint)  # Normalize

				# Check for parallelism
				dot_product = np.dot(vector_to_endpoint, vector_to_midpoint)
				if np.abs(dot_product) < 0.1:  # Threshold for parallelism
					# print(f"Vectors not parallel for endpoint {endpoint}. Discarding triplet.")
					parallel = False
					break

			if not parallel:
				continue  # Skip this triplet

			# add the midpoint to the points array
			points = np.vstack((points, midpoint))
			# add the midpoint to the endpoints
			endpoints.append(len(points) - 1)
			# connect the midpoint to the three endpoints by adding an extra row and column to the adjacency matrix
			adjacency_matrix = np.pad(adjacency_matrix, ((0, 1), (0, 1)), mode='constant')
			adjacency_matrix[-1, -1] = 0



			for i in range(3):
				adjacency_matrix[-1, triplet[i]] = 1
				adjacency_matrix[triplet[i], -1] = 1
			used_index.extend([triplet[0], triplet[1], triplet[2], len(points) - 1])
			#also add the midpoints to the used index
			# remove the triplets from the endpoints
			endpoints.remove(triplet[0])
			endpoints.remove(triplet[1])
			endpoints.remove(triplet[2])
	else: 
		# print('Too many triplet combinations to calculate in decent time. Are you sure you want to continue?')
		exit()

	return adjacency_matrix, endpoints, points, used_index


def connect_single_endpoints(adjacency_matrix, endpoints, points, used_index):
	# now connect the endpoints that are not part of a triplet
	# do this by connecting each endpoint to the point that is closest in the direction of the regression line
	# of the two closest points

	for endpoint in endpoints:
		# Make the vector from the point just before the endpoint to the endpoint by extracting the point just before the endpoint from the adjacency matrix
		# and subtracting the endpoint from that point
		previous_point = np.where(adjacency_matrix[endpoint] == 1)[0][0]
		direction_vector = points[endpoint] - points[previous_point]
		direction_vector = direction_vector / np.linalg.norm(direction_vector)
		# now take steps in the direction of the direction vector and find the closest point
		for i in range(1,15):
			new_point = points[endpoint] + direction_vector * i
			distances = np.linalg.norm(points - new_point, axis=1)
			# set the distance to the endpoint to a very large number
			distances[endpoint] = 1e9
			closest_point = np.argmin(distances)
			# check if the distance is below some threshold while avoiding connecting to the same point
			if distances[closest_point] < 2:
				''''
				# make a midpoint between the endpoint and the closest point
				# connect the endpoint to the midpoint by adding a row and column to the adjacency matrix
				adjacency_matrix = np.pad(adjacency_matrix, ((0, 1), (0, 1)), mode='constant')
				adjacency_matrix[-1, -1] = 0
				adjacency_matrix[-1, endpoint] = 1
				adjacency_matrix[endpoint, -1] = 1
				# also connect the midpoint to the closest point
				adjacency_matrix[-1, closest_point] = 1
				adjacency_matrix[closest_point, -1] = 1

				# add the midpoint to the points array after calculating the midpoint
				midpoint = (points[endpoint] + points[closest_point]) / 2
				points = np.vstack((points, midpoint))
				print('Connected endpoint')
				# remove the endpoint from the endpoints
				endpoints.remove(endpoint)
				# add the endpoint to the used index as well as the closest point and the midpoint
				used_index.extend([endpoint, closest_point, len(points) - 1])
				'''
				# connect the endpoint to the closest point
				adjacency_matrix[endpoint, closest_point] = 1
				adjacency_matrix[closest_point, endpoint] = 1
				used_index.extend([endpoint, closest_point])

				break

			# if i == 4:
				# print('No connection found for endpoint')
		# print('endpoints left:', len(endpoints))
	# G = nx.from_numpy_array(adjacency_matrix)
	# is_connected = nx.is_connected(G) 
	# if is_connected:
		# print("Graph is connected:", is_connected)
	# else:
		# print("Graph is connected:", is_connected)
		# components = [len(c) for c in nx.connected_components(G)]  # Get the number of nodes in each connected component
		# print("Number of connected components:", len(components), "with sizes:", components)
		# get the adjacency matrix and points of the largest connected component
		#largest_component = max(nx.connected_components(G), key=len)
		#indices = np.array(list(largest_component))
		#adjacency_matrix = adjacency_matrix[indices][:, indices]
		#points = points[indices]

	return adjacency_matrix, endpoints, points, used_index
def collapse_loops(adj_matrix, points):
	"""
	Collapse closed loops in the graph into a single center point.

	Parameters:
	- adj_matrix: numpy.ndarray, adjacency matrix of the graph.
	- points: numpy.ndarray, 3D coordinates of the points.

	Returns:
	- updated_adj_matrix: numpy.ndarray, updated adjacency matrix after collapsing loops.
	- updated_points: numpy.ndarray, updated 3D coordinates of the points.
	"""
	from scipy.spatial.distance import pdist, squareform
	# print("Adjacency matrix shape:", adj_matrix.shape, "Points shape:", points.shape)
	def detect_cycles(adj_matrix):
		"""Detect closed loops (cycles) in the graph."""
		from collections import defaultdict
		adj_list = defaultdict(list)
		for i in range(len(adj_matrix)):
			for j in range(i + 1, len(adj_matrix)):
				if adj_matrix[i, j] == 1:
					adj_list[i].append(j)
					adj_list[j].append(i)

		visited = set()
		cycles = []

		def dfs(node, parent, path):
			visited.add(node)
			path.append(node)
			for neighbor in adj_list[node]:
				if neighbor == parent:
					continue
				if neighbor in path:
					cycle_start = path.index(neighbor)
					cycles.append(path[cycle_start:])
				elif neighbor not in visited:
					dfs(neighbor, node, path)
			path.pop()

		for node in range(len(adj_matrix)):
			if node not in visited:
				dfs(node, -1, [])

		# Remove duplicates
		unique_cycles = []
		for cycle in cycles:
			sorted_cycle = sorted(cycle)
			if sorted_cycle not in unique_cycles:
				unique_cycles.append(sorted_cycle)
		return unique_cycles

	# Detect cycles
	cycles = detect_cycles(adj_matrix)

	if not cycles:
		# print("No cycles found.")
		return adj_matrix, points
	
	# print(f"Found {len(cycles)} cycle(s).")

	updated_points = points.tolist()
	updated_adj_matrix = adj_matrix.copy()

	for cycle in cycles:
		# Compute the center of the cycle
		cycle_points = points[cycle]
		center = np.mean(cycle_points, axis=0)
		center_idx = len(updated_points)
		updated_points.append(center)

		# Update adjacency matrix
		for node in cycle:
			# Remove connections within the cycle
			updated_adj_matrix[node, :] = 0
			updated_adj_matrix[:, node] = 0

		# Connect the center to all external neighbors of the cycle
		neighbors_to_connect = set()
		for node in cycle:
			neighbors = np.where(adj_matrix[node] == 1)[0]
			for neighbor in neighbors:
				if neighbor not in cycle:
					neighbors_to_connect.add(neighbor)

		# Update adjacency matrix for the new center
		updated_adj_matrix = np.pad(updated_adj_matrix, ((0, 1), (0, 1)), mode='constant')
		for neighbor in neighbors_to_connect:
			updated_adj_matrix[center_idx, neighbor] = 1
			updated_adj_matrix[neighbor, center_idx] = 1
	# print("Adjacency matrix shape:", updated_adj_matrix.shape, "Points shape:", len(updated_points))

	return updated_adj_matrix, np.array(updated_points)

def remove_isolated_points(adj_matrix, points):
	# Identify non-isolated points (rows or columns with non-zero entries)
	# print("Adjacency matrix shape:", adj_matrix.shape)
	# print("Points array shape:", points.shape)

	connected_mask = np.any(adj_matrix > 0, axis=1)

	# Filter out isolated points
	updated_adj_matrix = adj_matrix[connected_mask][:, connected_mask]
	updated_points = points[connected_mask]
	# print("Updated adjacency matrix shape:", updated_adj_matrix.shape)
	# print("Updated points array shape:", updated_points.shape)

	return updated_adj_matrix, updated_points

def post_processing(adjacency_matrix, points, x_points, used_index, closed_loop_removal=True):
	# Remove connections that passes through boundary points (x_points)
	# Also make sure that there are no smaller closed subgraphs
		# Create a KDTree for the boundary points
	boundary_tree = spatial.cKDTree(x_points)

	# Iterate over all pairs of used-to-be-endpoints
	for i in range(len(used_index)):
		for j in range(i + 1, len(used_index)):
			u = used_index[i]
			v = used_index[j]

			# Check if there is a connection between u and v
			if adjacency_matrix[u, v] == 1:
				# Define the line segment between u and v
				segment_start = points[u]
				segment_end = points[v]
				segment_vector = segment_end - segment_start
				segment_length = np.linalg.norm(segment_vector)

				# Generate sample points along the line segment
				t_values = np.linspace(0, 1, int(segment_length * 10))  # Adjust resolution as needed
				sampled_points = segment_start + np.outer(t_values, segment_vector)

				# Check if any sampled point is within a small radius of the boundary points
				for sampled_point in sampled_points:
					if len(boundary_tree.query_ball_point(sampled_point, 1)) > 0:  # Radius is adjustable
						# If the segment crosses a boundary, remove the connection
						adjacency_matrix[u, v] = 0
						adjacency_matrix[v, u] = 0
						# print(f"Removed connection between {u} and {v} (crosses boundary).")
						break

	if closed_loop_removal==True:
		# Collapse closed loops in the graph
		adjacency_matrix, points = collapse_loops(adjacency_matrix, points)

	# delete the points that are not connected to any other points
	adjacency_matrix, points = remove_isolated_points(adjacency_matrix, points)
		

	G = nx.from_numpy_array(adjacency_matrix)
	is_connected = nx.is_connected(G) 
	if is_connected:
		print("Graph is connected:", is_connected)
		print("Size of the graph:", len(G))
		adjacency_matrix_largest = adjacency_matrix
		points_largest = points
	else:
		print("Graph is connected:", is_connected)
		components = [len(c) for c in nx.connected_components(G)]  # Get the number of nodes in each connected component
		print("Number of connected components:", len(components), "with sizes:", components)
		# get the adjacency matrix and points of the largest connected component
		largest_component = max(nx.connected_components(G), key=len)
		indices = np.array(list(largest_component))
		adjacency_matrix_largest = adjacency_matrix[indices][:, indices]
		points_largest = points[indices]

	return adjacency_matrix_largest, points_largest, adjacency_matrix, points, is_connected
		




def visualize_graph_3d(adj_matrix, points_flattened, x_points=None):
	"""
	Visualizes a 3D graph using Plotly, coloring each subgraph with a unique color.
	
	Parameters:
		adj_matrix (numpy.ndarray): Adjacency matrix representing the graph.
		points_flattened (numpy.ndarray): Flattened array of 3D points with shape (n, 3),
										where n is the number of nodes.
		x_points (numpy.ndarray, optional): Additional points to visualize, shape (m, 3).
	"""
	# Extract x, y, z coordinates
	x, y, z = points_flattened[:, 0], points_flattened[:, 1], points_flattened[:, 2]

	# Identify connected components
	n_components, labels = connected_components(csgraph=adj_matrix, directed=False)

	# Generate unique colors for each subgraph
	colors = np.random.rand(n_components, 3)  # RGB colors
	node_colors = [f'rgb({c[0] * 255},{c[1] * 255},{c[2] * 255})' for c in colors[labels]]

	# Create edge traces
	edge_x, edge_y, edge_z, edge_colors = [], [], [], []
	for i in range(len(adj_matrix)):
		for j in range(i + 1, len(adj_matrix)):
			if adj_matrix[i, j] == 1:
				edge_x.extend([x[i], x[j], None])  # None to break the line
				edge_y.extend([y[i], y[j], None])
				edge_z.extend([z[i], z[j], None])
				edge_colors.append(node_colors[i])  # Use the starting node's color

	# Create the graph
	fig = go.Figure()

	# Add edge traces
	fig.add_trace(go.Scatter3d(
		x=edge_x, y=edge_y, z=edge_z,
		mode='lines',
		line=dict(color='rgba(1, 0, 0, 1)', width=5),
		name='Edges'
	))

	# Add node traces with unique colors for each component
	fig.add_trace(go.Scatter3d(
		x=x, y=y, z=z,
		mode='markers',
		marker=dict(size=5, color=node_colors),
		name='Nodes'
	))

	if x_points is not None:
		fig.add_trace(go.Scatter3d(
			x=x_points[:, 0], 
			y=x_points[:, 1], 
			z=x_points[:, 2],
			mode='markers',
			marker=dict(
				color='rgba(128, 128, 128, 0.8)',  # Grey color with 50% transparency
				size=2
			),
			name='Extra Points'
		))

	# Set layout
	fig.update_layout(
		scene=dict(
			xaxis_title='X Axis',
			yaxis_title='Y Axis',
			zaxis_title='Z Axis'
		),
		title="3D Graph Visualization",
		showlegend=True
	)

	fig.show()

def get_adjacency_matrix_and_points(folder):
	"""
	Load the adjacency matrix and points from an npy file.

	Parameters:"

	- path (str): Path to the npy file containing the adjacency matrix and points.

	Returns:

	- adjacency_matrix (numpy.ndarray): The adjacency matrix of the graph.
	- points (numpy.ndarray): The 3D coordinates of the points.
	"""
	# Load the data
	x, p = load_tree(folder, frame=-1)
	p = p / np.linalg.norm(p, axis=1)[:, None]
	
	#Variables
	threshold = 1.5
	number_of_points = len(x)
	indices = range(number_of_points)

	_, midpoints = closest_point_on_lines(x, p, threshold, indices)

	# Now we thin the line
	thinned_points, regression_lines = thin_line(midpoints, point_cloud_thickness=5, iterations=2, sample_points=0)

	sorted_points_collected = sort_points_in_segments(thinned_points, regression_lines, sorted_points_distance=3)

	adjacency_matrix1, endpoints, points = create_adjacency_matrix(sorted_points_collected)
	adjacency_matrix2, endpoints, points, used_index = connect_triplet_endpoints(adjacency_matrix1, endpoints, points, radius=5)
	adjacency_matrix3, endpoints, points, used_index = connect_single_endpoints(adjacency_matrix2, endpoints, points, used_index)
	_, _, adjacency_matrix4, points, is_connected = post_processing(adjacency_matrix3, points, x, used_index, closed_loop_removal=False)

	return adjacency_matrix4, points, is_connected

def quick_visualize_3D_graph(adj_mat, points):
    _, ax = plt.subplots(1,1, figsize=(10, 7), tight_layout=True, subplot_kw=dict(projection='3d'))
    ax.set_axis_off()
    color_lst_RGB = [(0, 0, 1), (1, 0, 0), (1, 0.8, 0), (0, 1, 0), (0.5, 0, 0.5), (1, 1, 0), (1, 0, 1), (0.5, 0.5, 0.5), (0, 0, 0), (0.5, 0.5, 0)]

    components = connected_components(adj_mat)[1]
    colors = np.zeros((len(points), 3))
    for i,comp in enumerate(np.unique(components)):
        colors[comp == components] = color_lst_RGB[i]

    #alpha scaling with distance
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=50, c=colors, depthshade=True )
    # set limits by range of data
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_zlim(z_min,z_max)
    ax.axes.set_aspect('equal')


    #plot the connections
    for i in range(len(adj_mat)):
        for j in range(i, len(adj_mat)):
            if adj_mat[i, j] == 1:
                ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], [points[i, 2], points[j, 2]], c='black', alpha=0.5)

    return None

if __name__ == '__main__':

	# Load the data
	# Testrun
	angle = 10
	seed  = 3 
	folder = f"vitro_tests/angle_fusion_tests/angle_{angle}_seed_{seed}/data.npy"

	x, p = load_tree(folder, frame=-1)
	p = p / np.linalg.norm(p, axis=1)[:, None]

	threshold = 1.5
	number_of_points = len(x)
	indices = range(number_of_points)

	results, midpoints = closest_point_on_lines(x, p, threshold, indices)

	# Now we thin the line
	thinned_points, regression_lines = thin_line(midpoints, point_cloud_thickness=5, iterations=2,sample_points=0)

	sorted_points_collected = sort_points_in_segments(thinned_points, regression_lines, sorted_points_distance=3)

	adjacency_matrix1, endpoints, points = create_adjacency_matrix(sorted_points_collected)

	adjacency_matrix2, endpoints, points, used_index = connect_triplet_endpoints(adjacency_matrix1, endpoints, points, radius=5)

	adjacency_matrix3, endpoints, points, used_index = connect_single_endpoints(adjacency_matrix2, endpoints, points, used_index)

	adjacency_matrix_largest, points_largest, adjacency_matrix4, points, is_connected = post_processing(adjacency_matrix3, points, x, used_index, closed_loop_removal=False)

	## Extract the directory path without the last subfolder
	#directory_path = os.path.dirname(path)

	## Save the points and the adjacency matrix as an npy file with the same name as the original file except the last part
	#output_path = os.path.join(directory_path, 'adjacency_matrix_and_points.npy')
	#output_path = output_path.replace('\\', '/')
	#saved_dict = {'adjacency_matrix': adjacency_matrix4, 'points': points}
	#np.save(output_path, saved_dict, allow_pickle=True)


	visualize_graph_3d(adjacency_matrix4, points, x_points=x)

	# Visualize the results
	canvas = scene.SceneCanvas(keys='interactive', show=True)
	view = canvas.central_widget.add_view()


	# Generate line segments from adjacency matrix
	line_segments = []
	for i in range(adjacency_matrix4.shape[0]):
		for j in range(i + 1, adjacency_matrix4.shape[1]):  # Check upper triangle to avoid duplicates
			if adjacency_matrix4[i, j] > 0:  # Connection exists
				line_segments.append(points[i])  # Start of line
				line_segments.append(points[j])  # End of line

	# Convert to a NumPy array
	line_segments = np.array(line_segments)

	# Add all lines as a single visual
	line_visual = visuals.Line()
	line_visual.set_data(line_segments, connect='segments', color='red', width=5)
	view.add(line_visual)

	index_color = np.random.rand(len(sorted_points_collected), 3)

	for idx, sorted_points in enumerate(sorted_points_collected):
		sorted_marker = visuals.Markers(spherical=True, scaling=True)
		sorted_marker.set_data(sorted_points, face_color=index_color[idx], size=1.2, edge_width=0)
		view.add(sorted_marker)


	structure = visuals.Markers(spherical=True, scaling=True)
	structure.set_data(x, face_color=[1,1,1, 1], size=1.2, edge_width=0)
	view.add(structure)


	view.camera = 'fly'
	view.camera.move_speed = 0.1  # Reduce movement speed

	## Run the Vispy application
	canvas.app.run()

