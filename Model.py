from difflib import SequenceMatcher

import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

global_method = "complete"
global_metric = "euclidean"


# Elbow Method
def calculate_gradient(coord1, coord2):
    x1 = coord1[0]
    y1 = coord1[1]
    x2 = coord2[0]
    y2 = coord2[1]
    gradient = (y1 - y2) / (x1 - x2)
    return gradient


def calculate_y_intercept(coord1, m):
    x1 = coord1[0]
    y1 = coord1[1]
    c = m * (-x1) + y1
    return c


def arch_linear_line(coord1, coord2):
    m = calculate_gradient(coord1, coord2)
    c = calculate_y_intercept(coord1, m)
    return tuple((m, c))


def cal_euclidean_distance(coord1, coord2):
    x1 = coord1[0]
    y1 = coord1[1]
    x2 = coord2[0]
    y2 = coord2[1]
    return (((y2 - y1) ** 2) + ((x2 - x1) ** 2)) ** 0.5


def cal_x_point_contact(linear_c, linear_m, c, m):
    return (c + linear_c) / (linear_m - m)


def calculate_point_c_of_linear_line(coord, linear_c, linear_m):
    perpendicular_m = -1 / linear_m
    coord_c = calculate_y_intercept(coord, perpendicular_m)
    x_point_contact = cal_x_point_contact(linear_c, linear_m, coord_c, perpendicular_m)
    y_point_contact = (linear_m * x_point_contact) + linear_c
    return cal_euclidean_distance(coord, (x_point_contact, y_point_contact))


def calculate_elbow_method(list_ssd: list):
    index = 1
    if len(list_ssd) > 2:
        m_c = arch_linear_line((2, list_ssd[0]), (2 + len(list_ssd), list_ssd[len(list_ssd) - 1]))
        m = m_c[0]
        c = m_c[1]
        list_of_distance = []
        for i in range(1, len(list_ssd) - 1):
            list_of_distance.append((i + 2, calculate_point_c_of_linear_line((i + 2, list_ssd[i]), c, m)))
        df = pd.DataFrame.from_records(list_of_distance, columns=["record", "ssd"])
        index = df["record"].iat[0]
        mid = len(list_ssd) // 2
        if 3 < mid and mid > index:
            print("WARNING: Clustering might not be accurate")
    return index


# TF-IDF

def tfidf_transformation(df):
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3))
    np_array = tfidf.fit_transform(df["content"])
    linkage_data = linkage(np_array.toarray(), method=global_method, metric=global_metric)
    dendrogram(linkage_data)
    plt.show()


def tfidf_kmeans_transformation(df, location):
    print("Logs: tfidf - " + location)
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3))
    np_array = tfidf.fit_transform(df["content"])
    sum_of_squared_distances = []
    K = range(2, df.shape[0])
    for k in K:
        km = KMeans(n_clusters=k, max_iter=500, n_init=10, random_state=2)
        km = km.fit(np_array.toarray())
        sum_of_squared_distances.append(km.inertia_)
        if km.inertia_ == 0:
            break
    num_of_k = calculate_elbow_method(sum_of_squared_distances)
    kmeans_model = KMeans(n_clusters=num_of_k, max_iter=500, n_init=10, random_state=2)
    kmeans_model = kmeans_model.fit(np_array.toarray())
    plt.plot(range(2, len(sum_of_squared_distances) + 2), sum_of_squared_distances, "bx-")
    plt.xlabel("k")
    plt.ylabel("Sum_of_squared_distances")
    plt.title("Elbow Method For Optimal k")
    plt.savefig(location + 'tfidfk')
    plt.show()
    return kmeans_model


# Similarity
def list_of_list(df):
    overall_list = []
    target_list = df["content"].tolist()

    for i in target_list:
        list_content = []
        for ii in target_list:
            sm = SequenceMatcher(
                a=i, b=ii
            )
            list_content.append(sm.ratio())
        overall_list.append(list_content)
    return overall_list


def similarity_hierarchy_transformation(df):
    autoscaler = StandardScaler()
    linkage_data = linkage(autoscaler.fit_transform(list_of_list(df)), method=global_method, metric=global_metric)
    dendrogram(linkage_data)
    plt.show()


def similarity_k_means(df, location):
    print("Logs: similarity - " + location)
    autoscaler = StandardScaler()
    features = autoscaler.fit_transform(list_of_list(df))
    sum_of_squared_distances = []
    K = range(2, df.shape[0])
    for k in K:
        km = KMeans(n_clusters=k, max_iter=500, n_init=10, random_state=2)
        km = km.fit(features)
        if km.inertia_ == 0:
            break
        else:
            sum_of_squared_distances.append(km.inertia_)
    num_of_k = calculate_elbow_method(sum_of_squared_distances)
    kmeans_model = KMeans(n_clusters=num_of_k, max_iter=500, n_init=10, random_state=2)
    kmeans_model = kmeans_model.fit(features)
    plt.plot(range(2, len(sum_of_squared_distances) + 2), sum_of_squared_distances, "bx-")
    plt.xlabel("k")
    plt.ylabel("Sum_of_squared_distances")
    plt.title("Elbow Method For Optimal k")
    plt.savefig(location + 'sk')
    plt.show()
    return kmeans_model
