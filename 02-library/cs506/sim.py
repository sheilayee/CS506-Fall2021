from scipy import spatial

def euclidean_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i])**2
    return res**(1/2)

def manhattan_dist(x, y):
    return sum(abs(x_num-y_num) for x_num, y_num in zip(x,y))

def jaccard_dist(x, y):
    intersection = len(list(set(x).intersection(y)))
    union = (len(x) + len(y)) - intersection
    return float(intersection) / union

def cosine_sim(x, y):
    result = 1 - spatial.distance.cosine(x, y)

# Feel free to add more
