from midiutil import MIDIFile
import random
# Major chords
a_flat = [56,60,63]
b_flat = []
c_maj = [48,52,55]
f_maj = [53,57,60]
g_maj = [55,59,62]

# Minor chords
a_min = [57,60,64]
b_flat_min = [58,62,65]
c_min = [48,51,55]
d_min = [50,53,56]
e_min = [52,55,59]
f_min = [53,56,60]
g_min = [55,58,62]

# 2-5-1 Jazz chord progression
d_maj7 = [50,53,57,48]
g_maj7 = [55,59,50,53]
c_maj7 = [48,52,55,59]

chord_prog_pop = [c_maj,c_maj,a_min,a_min, f_maj,f_maj,g_maj,g_maj]

track    = 0
channel  = 0
time     = 0    # In beats
duration = 1    # In beats
tempo    = 120  # In BPM
volume   = 100  # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                      # automatically)

MyMIDI.addTempo(track, time, tempo)

# Creates .midi notes
for i, chord in enumerate(chord_prog_pop):
    for j, pitch in enumerate(chord):
        #random.randint(1,4)
        MyMIDI.addNote(track, channel, pitch, duration, volume)

# Writes .midi file
with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)