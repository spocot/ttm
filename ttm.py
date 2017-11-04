import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sentences = []

vowels = ['a','e','i','o','u','y']

with open("pp.txt", "r", encoding='utf-8') as f:
    content = f.read()
    sentences = [x.strip() for x in content.replace('\n',' ').replace(';','.').replace('_','').replace('"','.').replace('“','.').replace('”','.').split('.') if len(x.strip().split()) > 5]

def est_syl(word):
    if len(word) == 1:
        return 1
    word = word.lower()
    count = 0
    for i in range(len(word) - 1):
        if word[i] in vowels and (word[i+1] not in vowels or i == len(word)-2):
            count += 1
    return count

def mk_datapoint(sentence):
    sentence = sentence.split()
    words = len(sentence)
    syls = sum([est_syl(word) for word in sentence])
    avg = np.mean(list(map(lambda x: len(x), sentence)))
    return [words,avg,syls]

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def show_plot(nclusters, datapoints):
    df = pd.DataFrame(data=np.array(datapoints))

    kmeans = KMeans(n_clusters=nclusters)
    kmeans.fit(df)

    labels = kmeans.predict(df)

    centroids = kmeans.cluster_centers_
    cmap = get_cmap(nclusters)
    colmap = {
        i: cmap(i)
        for i in range(nclusters)
    }

    fig = plt.figure()
    colors = list(map(lambda x: colmap[x], labels))

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[0], df[1], df[2], color=colors, alpha=0.5, edgecolor='k')

    for idx, centroid in enumerate(centroids):
        ax.scatter(*centroid, color=colmap[idx])

    ax.set_xlabel('Num_Words')
    ax.set_ylabel('Avg_L')
    ax.set_zlabel('Num_Syl')

    plt.show()

    return df

def kcluster(nclusters, datapoints):

    df = pd.DataFrame(data=np.array(datapoints))

    kmeans = KMeans(n_clusters=nclusters)
    kmeans.fit(df)

    labels = kmeans.predict(df)

    centroids = kmeans.cluster_centers_

    return list(labels)

chords = kcluster(6, list(map(lambda x: mk_datapoint(x), sentences)))

d = {
    i: 0
    for i in range(6)
}

for c in chords:
    d[c] += 1

print(np.random.choice(list(d.keys()), 200, p=[x / len(sentences) for x in list(d.values())]))