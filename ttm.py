import numpy as np
import pandas as pd
import skfuzzy as fuzz
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from faker import Faker

fake = Faker()

sentences = []

vowels = ['a','e','i','o','u','y']

for i in range(10):
    sentences.append(fake.sentence())

with open("pp.txt", "r", encoding='utf-8') as f:
    content = f.read()
    sentences = [x.strip() for x in content.replace('\n',' ').replace(';','.').replace('_','').replace('"','.').replace('“','.').replace('”','.').split('.') if len(x.strip().split()) > 5]
    for l in sentences:
        print(l)

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

datapoints = list(map(lambda x: mk_datapoint(x), sentences))
for dp in datapoints:
    print(dp)

print(len(datapoints))

num_words = [x[0] for x in datapoints]
avg_l = [x[1] for x in datapoints]
num_syl = [x[2] for x in datapoints]

min_num_words = min(num_words)
max_num_words = max(num_words)
min_avg_l = min(avg_l)
max_avg_l = max(avg_l)
min_num_syl = min(num_syl)
max_num_syl = max(num_syl)

#np.random.seed(123)
k = 4

centroids = {
    i+1: [np.random.randint(min_num_words, max_num_words+1),
          np.random.ranf() * max_avg_l + min_avg_l,
          np.random.randint(min_num_syl, max_num_syl+1)]
    for i in range(k)
}

C = np.array(datapoints)
df = pd.DataFrame(data=C)
print(df)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df[0], df[1], df[2], color='k')

colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'r'}
for i in centroids.keys():
    ax.scatter(*centroids[i], color=colmap[i])

ax.set_xlabel('Num_Words')
ax.set_ylabel('Avg_L')
ax.set_zlabel('Num_Syl')

plt.show()

#print(est_syl("vial"))
#print(sentences)
#print(list(map(lambda x: list(map(lambda y: est_syl(y), x.split())), sentences)))
#print(list(map(lambda x: list(map(lambda y: nsyl(y), x.split())), sentences)))