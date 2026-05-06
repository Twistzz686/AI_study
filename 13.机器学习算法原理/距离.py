import math

x = [1,2]
y = [4,6]


# 欧式距离
def eucalidean_siatance(x,y):
    return math.sqrt(sum([(a-b) ** 2 for a,b in zip(x,y)]))

print("欧氏距离",eucalidean_siatance(x,y))

# 曼哈顿距离
def manhattan_distance(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))

print("曼哈顿距离",manhattan_distance(x,y))

# 切比雪夫距离
def chebyshev_distance(x,y):
    return max([abs(a-b) for a,b in zip(x,y)])

print("切比雪夫距离",chebyshev_distance(x,y))

# 余弦相似度
def cosine_similarity(x,y):
    numerator = sum(a * b for a,b in zip(x,y))
    denominator = math.sqrt(sum(a ** 2 for a in x)) * math.sqrt(sum(b ** 2 for b in y))
    return numerator / denominator

print("余弦相似度",cosine_similarity(x,y))

# 汉明距离
def hamming_distance(x_str,y_str):
    return sum(a !=b for a,b in zip(x_str,y_str))

x_str = "101100"
y_str = "111000"

print(hamming_distance(x_str,y_str))

# 闵可夫斯基距离
def minkovski_distance(x,y,p):
    return sum(abs(a-b) ** p for a,b in zip(x,y)) ** (1/p)

p=1
print("闵可夫斯基距离",minkovski_distance(x,y,p))

# jaccrd
def jaccard_index(x_set,y_set):
    intersection = len(set(x_set & y_set))
    union = len(set(x_set | y_set))
    return intersection / union

x_set = {1,2,3}
y_set = {2,3,4}

print("jaccrd",jaccard_index(x_set,y_set))

# 半正失距离
def haversine_distance(lat1,lon1,lat2,lon2):
    R = 6371.0

    lat1 = math.radians(lat1)
    lat1 = math.radians(lon1)
    lat1 = math.radians(lat2)
    lat1 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    d = 2 * R * math.asin(math.sqrt(math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2))
    return d

lat1,lon1 = 52.2296756,21.0122287
lat2,lon2 = 51.5073509,-0.1277583

print("半正失距离",haversine_distance(lat1,lon1,lat2,lon2))