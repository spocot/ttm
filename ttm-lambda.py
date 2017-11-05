import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from midiutil import MIDIFile

class Genre:
    """Class to define chords and write to output .mid file"""
    # Major chords
    a_flat = [56, 60, 63]
    b_flat = []
    c_maj = [48, 52, 55]
    f_maj = [53, 57, 60]
    g_maj = [55, 59, 62]

    # Minor chords
    a_min = [57, 60, 64]
    b_flat_min = [58, 62, 65]
    c_min = [48, 51, 55]
    d_min = [50, 53, 56]
    e_min = [52, 55, 59]
    f_min = [53, 56, 60]
    g_min = [55, 58, 62]

    # 2-5-1 Jazz chord progression
    d_maj7 = [50, 53, 57, 48]
    g_maj7 = [55, 59, 50, 53]
    c_maj7 = [48, 52, 55, 59]

    #chords to be used in the output
    chord_prog_pop = [c_maj, a_min, f_maj, g_maj, d_maj7, g_maj7, c_maj7]

    track = 0
    channel = 7
    time = 0  # In beats
    duration = 1  # In beats
    tempo = 120  # In BPM
    volume = 100  # 0-127, as per the MIDI standard

    def make_midi(self, chordprog, durations):
        """method to write to output file"""
        midi = MIDIFile(1)
        midi.addTempo(self.track, self.time, self.tempo)

        time = 0
        for i, chord in enumerate(chordprog):
            d = durations[i]
            for j, pitch in enumerate(chord):
                midi.addNote(self.track, self.channel, pitch, time, d, self.volume)
            time += d

        # Writes .midi file
        with open("od.mid", "wb") as output_file:
            midi.writeFile(output_file)

sentences = []

vowels = ['a','e','i','o','u','y']

#read in large text file, sanitize input
with open("odyssey.txt", "r", encoding='utf-8') as f:
    content = f.read()
    sentences = [x.strip() for x in content.replace('\n',' ').replace(';','.').replace(':','.').replace('_','').replace('"','.').replace('“','.').replace('”','.').split('.') if len(x.strip().split()) > 5]

#method to estimate the number of syllables in a word (accurate to +/- 1 syllable)
def est_syl(word):
    if len(word) == 1:
        return 1
    word = word.lower()
    count = 0
    for i in range(len(word) - 1):
        if word[i] in vowels and (word[i+1] not in vowels or i == len(word)-2):
            count += 1
    return count

#analyzes the three datapoints of a sentence: len, avg word len, num syllables
def mk_datapoint(sentence):
    sentence = sentence.split()
    words = len(sentence)
    syls = sum([est_syl(word) for word in sentence])
    avg = np.mean(list(map(lambda x: len(x), sentence)))
    return [words,avg,syls]

#colormap function for plotting
def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

#display a nice looking plot of all the clusters
def show_plot(nclusters, df):

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

#k-means clustering
def kcluster(nclusters, df):

    kmeans = KMeans(n_clusters=nclusters)
    kmeans.fit(df)

    labels = kmeans.predict(df)

    centroids = kmeans.cluster_centers_

    return list(labels)

cs = Genre.chord_prog_pop

dpoints = list(map(lambda x: mk_datapoint(x), sentences))
df = pd.DataFrame(data=np.array(dpoints))

show_plot(len(cs), df)

chords = kcluster(len(cs), df)

d = {
    i: 0
    for i in range(len(cs))
}
#print(np.max(dpoints[0]))
durations = [dp for dp in df[0]]
#durations = list(map(lambda x: map_length(x), df[0]))

min_d = np.min(durations)
max_d = np.max(durations)

#choose duration of each chord as a function of sentence length
durations = [round(4 * (d - min_d)/(max_d - min_d))/4 * 7 + 0.5 for d in durations]
print(durations)
for c in chords:
    d[c] += 1

#song = np.random.choice(list(d.keys()), 300, p=[x / len(sentences) for x in list(d.values())])
l = list(range(len(chords)))
plt.scatter(l,chords)
plt.plot(l, chords)
plt.show()

chord_prog = [cs[i] for i in chords]
g = Genre()

#create the output music file
g.make_midi(chord_prog, durations)
