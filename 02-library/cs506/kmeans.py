from collections import defaultdict
from math import inf
import random
import numpy as np
import csv


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)
    
    Returns a new point which is the center of all the points.
    """
    means = []
    n = len(points)

    for col in range(len(points[0])):
        total = 0
        for row in range(len(points)):
            total += points[row][col]
        means.append(total / n)
    return means


def update_centers(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers in a list
    """
    clusters = list(set(assignments))
    clusters.sort()
    updated_centers = []

    for c in clusters:
        points_in_each_clusters = []

        for i in range(len(dataset)):
            # sort points by cluster

            if assignments[i] == c:
                points_in_each_clusters.append(dataset[i])

        # find average point in the cluster
        updated_centers.append(point_avg(points_in_each_clusters))

    return updated_centers

def assign_points(data_points, centers):
    """
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    return np.sqrt(np.sum((a - b) ** 2))

def distance_squared(a, b):

    return distance(a, b) ** 2

def generate_k(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """

    # randomly select the indices to be used to determine k points
    indices = random.shuffle([*range(len(dataset))])
    k_points = indices[:k]
    initial_points = []

    for index in k_points:
        initial_points.append(dataset[index])

    return initial_points

def cost_function(clustering):

    cost = 0 # the sum of squares
    for cluster in clustering.values():
        avg = point_avg(cluster)
        for point in cluster:
            cost += distance_squared(point, avg)
    return cost


def generate_k_pp(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    where points are picked with a probability proportional
    to their distance as per kmeans pp
    """
    # determine what the first centroid is
    num_points = len(dataset)
    first_idx = random.choice(range(num_points))
    centroids = [dataset[first_idx]]

    distances = [float("inf")] * num_points

    for points in range(k - 1):
        # loop through all points
        for point_ind in range(num_points):
            dist = distance(centroids[points], dataset[point_ind])
            if dist < distances[point_ind]:
                # append distances
                distances[point_ind] = distance

        max_val = max(distances)
        next_idx = distances.index(max_val)
        centroids.append(dataset[next_idx])

    return centroids

def _do_lloyds_algo(dataset, k_points):
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering


def k_means(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")
    
    k_points = generate_k(dataset, k)
    return _do_lloyds_algo(dataset, k_points)


def k_means_pp(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k_pp(dataset, k)
    return _do_lloyds_algo(dataset, k_points)
