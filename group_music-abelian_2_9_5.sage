import numpy as np

# Falls du midiutil benutzt, ist der Import oft:
# from midiutil import MIDIFile
#
# In deinem Original war es:
# from MidiFile import MIDIFile
#
# Unten bleibt es bei deiner Variante.
# Falls das bei dir nicht läuft, ersetze:
# from MidiFile import MIDIFile
# durch:
# from midiutil import MIDIFile


def writePitches(
    fn,
    inds,
    tempo=82,
    instrument=None,
    add21=True,
    start_at=None,
    durationsInQuarterNotes=False
):
    from MidiFile import MIDIFile

    ni = len(inds)

    if instrument is None:
        instrument = ni * [0]

    if start_at is None:
        start_at = ni * [0]

    if len(instrument) < ni:
        instrument = instrument + (ni - len(instrument)) * [0]

    if len(start_at) < ni:
        start_at = start_at + (ni - len(start_at)) * [0]

    MyMIDI = MIDIFile(ni, adjust_origin=False)

    for k in range(ni):
        MyMIDI.addTempo(k, 0, tempo)
        MyMIDI.addProgramChange(k, k, 0, instrument[k])

    times = list(start_at)

    for k in range(ni):
        channel = k
        track = k

        for event in inds[k]:

            pitch, duration, volume, isPause, event_tempo = event

            if not durationsInQuarterNotes:
                duration = 4 * duration

            if not isPause:
                pitch = int(np.floor(pitch))

                if add21:
                    pitch += 21

                pitch = max(0, min(127, pitch))
                volume = max(0, min(127, int(volume)))

                MyMIDI.addNote(
                    track,
                    channel,
                    int(pitch),
                    float(times[k]),
                    float(duration),
                    int(volume)
                )

            times[k] += float(duration)

    with open(fn, "wb") as output_file:
        MyMIDI.writeFile(output_file)

    print("written:", fn)


def group_music(
    G,
    chords,
    tempo=81,
    base_duration=1/9,
    octave_range=(-2, 2),
    pi_nr = pi,
    pi_digits=10000,
    voices=4,
    chord_change_every=4,
    iter_it = 1
):
    """
    Erzeugt musikalisches Material aus einer beliebigen endlichen SageMath-Gruppe G.

    G:
        Endliche SageMath-Gruppe, z.B.
        SymmetricGroup(4),
        DihedralGroup(8),
        AlternatingGroup(5),
        CyclicPermutationGroup(12),
        AbelianGroup([4, 4])

    chords:
        Liste von Akkorden als MIDI-Pitches.

    Rückgabe:
        Liste von Stimmen im Format:
        [
            [(pitch, duration, volume, isPause, tempo), ...],
            [(pitch, duration, volume, isPause, tempo), ...],
            ...
        ]
    """

    elts = list(G)
    n = len(elts)

    if iter_it > 1:
        my_elts = []
        for k in range(iter_it-1):          
            for g in elts:
                orbit = [g*h for h in elts]
                my_elts.extend(orbit)
        elts = my_elts        

    idx = {g: i for i, g in enumerate(elts)}

    rndbits = [int(x) % 2 for x in str(pi_nr.n(pi_digits))[2:]]

    lines = [[] for _ in range(voices)]

    octaves = []
    signs = []
    durations = []

    for v in range(voices):
        octave = -2 + (v % 5)
        octaves.append(octave)

        signs.append(1)

        durations.append(base_duration / (1 + (v % 4)))

    lastbit = 0
    c = 0

    for g in elts:

        # Reguläre Wirkung:
        # Jedes Gruppenelement g permutiert die Gruppe durch Linksmultiplikation.
        orbit = [g * h for h in elts]

        chord = chords[(c // chord_change_every) % len(chords)]
        m = len(chord)

        bit = rndbits[c % len(rndbits)]

        for j, h in enumerate(orbit):

            x = h
            element_index = idx[x]

            for v in range(voices):

                # Verschiedene Stimmen greifen unterschiedlich in den Akkord.
                if v % 4 == 0:
                    degree = (element_index + v) % m
                elif v % 4 == 1:
                    degree = (-element_index - 1 + v) % m
                elif v % 4 == 2:
                    degree = (2 * element_index + v) % m
                else:
                    degree = (element_index * element_index + v) % m

                pitch = chord[degree] + 12 * octaves[v]

                while pitch < 0:
                    pitch += 12

                while pitch > 127:
                    pitch -= 12

                # Dynamik pro Stimme unterschiedlich.
                if v % 2 == 0:
                    volume = 90 - int(50 * j / max(1, n - 1))
                else:
                    volume = 45 + int(50 * j / max(1, n - 1))

                volume = volume - 4 * (v // 2)
                volume = max(1, min(127, volume))

                duration = durations[v]

                lines[v].append((
                    int(pitch),
                    float(duration),
                    int(volume),
                    False,
                    tempo
                ))

        # Oktavbewegungen aktualisieren.
        for v in range(voices):

            if octaves[v] <= octave_range[0]:
                signs[v] = 1

            if octaves[v] >= octave_range[1]:
                signs[v] = -1

            # Verschiedene Stimmen reagieren unterschiedlich auf Pi-Bits.
            if v % 3 == 0:
                trigger = bit == 0
            elif v % 3 == 1:
                trigger = bit == 1
            else:
                trigger = lastbit != bit

            if trigger:
                octaves[v] = min(
                    max(octaves[v] + signs[v], octave_range[0]),
                    octave_range[1]
                )

        # Rhythmische Rotation statt nur Tausch zweier Stimmen.
        if bit == 1 and voices > 1:
            durations = durations[1:] + durations[:1]

        lastbit = bit
        c += 1

    return lines


chords = [
    # Moll / Dur Grundmaterial
    [62, 65, 69],          # d-Moll
    [65, 69, 72],          # F-Dur
    [67, 71, 74],          # G-Dur
    [60, 63, 67],          # c-Moll
    [60, 64, 67],          # C-Dur
    [57, 60, 64],          # a-Moll
    [55, 59, 62],          # G-Dur tief
    [53, 57, 60],          # F-Dur tief
    [50, 53, 57],          # d-Moll tief
    [52, 55, 59],          # e-Moll
    [59, 62, 65],          # h vermindert / B diminished
    [58, 62, 65],          # B-Dur / Bb-Dur

    # Septakkorde
    [60, 64, 67, 71],      # Cmaj7
    [62, 65, 69, 72],      # Dm7
    [64, 67, 71, 74],      # Em7
    [65, 69, 72, 76],      # Fmaj7
    [67, 71, 74, 77],      # G7
    [69, 72, 76, 79],      # Am7
    [59, 62, 65, 69],      # Bm7b5
    [58, 62, 65, 68],      # Bb7
    [61, 65, 68, 71],      # C#dim7-artig
    [63, 66, 69, 72],      # Eb dim7-artig

    # Moll mit Farbe
    [60, 63, 67, 70],      # Cm7
    [62, 65, 69, 72],      # Dm7
    [65, 68, 72, 75],      # Fm7
    [67, 70, 74, 77],      # Gm7
    [69, 72, 76, 79],      # Am7
    [57, 60, 64, 67],      # Am7 tief
    [50, 53, 57, 60],      # Dm7 tief
    [55, 58, 62, 65],      # Gm7 tief

    # Sus-Akkorde
    [60, 65, 67],          # Csus4
    [60, 62, 67],          # Csus2
    [62, 67, 69],          # Dsus4
    [62, 64, 69],          # Dsus2
    [65, 70, 72],          # Fsus4
    [65, 67, 72],          # Fsus2
    [67, 72, 74],          # Gsus4
    [67, 69, 74],          # Gsus2
    [69, 74, 76],          # Asus4
    [69, 71, 76],          # Asus2

    # Add9 / Add11 / Add6
    [60, 64, 67, 74],      # Cadd9
    [62, 65, 69, 76],      # Dm add9
    [65, 69, 72, 79],      # Fadd9
    [67, 71, 74, 81],      # Gadd9
    [57, 60, 64, 71],      # Am add9
    [60, 64, 67, 69],      # C6
    [62, 65, 69, 71],      # Dm6
    [65, 69, 72, 74],      # F6
    [67, 71, 74, 76],      # G6
    [60, 65, 67, 74],      # C sus4 add9

    # Maj9 / m9 / dominante Farben
    [60, 64, 67, 71, 74],  # Cmaj9
    [62, 65, 69, 72, 76],  # Dm9
    [64, 67, 71, 74, 78],  # Em9
    [65, 69, 72, 76, 79],  # Fmaj9
    [67, 71, 74, 77, 81],  # G9
    [69, 72, 76, 79, 83],  # Am9
    [58, 62, 65, 68, 72],  # Bb9
    [55, 59, 62, 65, 69],  # G13 ohne 9/11, kompakt

    # Vermindert / halbvermindert / übermäßig
    [60, 63, 66],          # Cdim
    [62, 65, 68],          # Ddim
    [64, 67, 70],          # Edim
    [65, 68, 71],          # Fdim
    [67, 70, 73],          # Gdim
    [60, 64, 68],          # C augmented
    [62, 66, 70],          # D augmented
    [65, 69, 73],          # F augmented
    [67, 71, 75],          # G augmented
    [60, 63, 66, 69],      # Cdim7
    [62, 65, 68, 71],      # Ddim7
    [64, 67, 70, 73],      # Edim7

    # Quartale / offene Voicings
    [60, 65, 70],          # Quartal C-F-Bb
    [62, 67, 72],          # Quartal D-G-C
    [64, 69, 74],          # Quartal E-A-D
    [65, 70, 75],          # Quartal F-Bb-Eb
    [67, 72, 77],          # Quartal G-C-F
    [69, 74, 79],          # Quartal A-D-G
    [60, 67, 74],          # offene Quinten C-G-D
    [62, 69, 76],          # offene Quinten D-A-E
    [65, 72, 79],          # offene Quinten F-C-G
    [67, 74, 81],          # offene Quinten G-D-A

    # Modale Farben
    [60, 62, 67, 69],      # C pentatonisch-artig
    [62, 65, 67, 69],      # D dorisch kompakt
    [64, 65, 71, 72],      # E phrygisch Farbe
    [65, 67, 72, 74],      # F lydisch/sus2 Farbe
    [67, 69, 74, 77],      # G mixolydisch Farbe
    [69, 72, 74, 79],      # A äolisch Farbe
    [59, 60, 65, 67],      # B lokrisch Farbe
    [60, 62, 65, 67, 69],  # C pentatonisch
    [62, 65, 67, 69, 72],  # D-Moll pentatonisch
    [65, 67, 69, 72, 74],  # F-Dur pentatonisch

    # Chromatischere / jazzigere Farben
    [60, 64, 67, 70, 74],  # C9
    [60, 64, 68, 70],      # C7#5
    [60, 64, 66, 70],      # C7b5
    [60, 63, 66, 70],      # Cm7b5
    [62, 66, 69, 72],      # D7
    [62, 66, 70, 72],      # D7#5
    [62, 66, 68, 72],      # D7b5
    [65, 69, 72, 75],      # F7
    [67, 71, 75, 77],      # G7#5
    [67, 71, 73, 77],      # G7b5
    [69, 73, 76, 79],      # A7
    [69, 72, 75, 79],      # Am7b5-ish

    # Cluster / dichte Texturen
    [60, 61, 64, 67],      # C mit kleiner Sekunde
    [62, 63, 65, 69],      # Dm mit b2
    [65, 66, 69, 72],      # F mit b2
    [67, 68, 71, 74],      # G mit b2
    [60, 62, 63, 67],      # C sus/add cluster
    [62, 64, 65, 69],      # D sus/add cluster
    [65, 67, 68, 72],      # F sus/add cluster
    [67, 69, 70, 74],      # G sus/add cluster

    # Tiefe, breite Voicings
    [48, 55, 60, 64, 67],  # C breit
    [50, 57, 62, 65, 69],  # Dm breit
    [53, 60, 65, 69, 72],  # F breit
    [55, 62, 67, 71, 74],  # G breit
    [57, 64, 69, 72, 76],  # Am breit
    [46, 53, 58, 62, 65],  # Bb breit
    [43, 50, 55, 59, 62],  # G breit tief
    [45, 52, 57, 60, 64],  # A breit tief

    # Hohe, gläserne Voicings
    [72, 76, 79],          # C-Dur hoch
    [74, 77, 81],          # Dm hoch
    [77, 81, 84],          # F-Dur hoch
    [79, 83, 86],          # G-Dur hoch
    [81, 84, 88],          # Am hoch
    [72, 76, 79, 83],      # Cmaj7 hoch
    [74, 77, 81, 84],      # Dm7 hoch
    [77, 81, 84, 88],      # Fmaj7 hoch
    [79, 83, 86, 89],      # G7 hoch
    [81, 84, 88, 91],      # Am7 hoch
]


def regular_permutation_group(G, side="left", return_data=True, key_func=None, check=True):
    """
    Wandelt eine endliche SageMath-Gruppe G in ihre reguläre Permutationsdarstellung um.

    Input:
        G           endliche Sage-Gruppe
        side        "left" oder "right"
                    "left":  x |-> a*x
                    "right": x |-> x*a
        return_data Falls True, werden zusätzlich Elemente, Index usw. zurückgegeben.
        key_func    optionale Funktion, um Gruppenelemente eindeutig zu hashen
        check       prüft, ob die Ordnung erhalten bleibt

    Output:
        Wenn return_data=False:
            PermutationGroup in SymmetricGroup(|G|)

        Wenn return_data=True:
            Dictionary mit:
                "group"        Permutationsgruppe
                "elements"     Liste der Gruppenelemente
                "index"        Element-Key -> Nummer 1..n
                "perm_of_gen"  Generator -> zugehörige Permutation
                "S"            SymmetricGroup(n)
                "key"          verwendete Key-Funktion
    """

    if side not in ["left", "right"]:
        raise ValueError("side muss 'left' oder 'right' sein.")

    elts = list(G)
    n = len(elts)
    S = SymmetricGroup(n)

    def default_key(x):
        """
        Robuster Key für viele Sage-Gruppenelemente:
        - hashbare Objekte direkt
        - Matrixgruppen über Matrixeinträge
        - sonst tuple(x)
        - sonst repr(x) als Fallback
        """
        try:
            hash(x)
            return ("object", x)
        except TypeError:
            pass

        if hasattr(x, "matrix"):
            M = x.matrix()
            return ("matrix", tuple(M.list()))

        try:
            t = tuple(x)
            hash(t)
            return ("tuple", t)
        except Exception:
            pass

        return ("repr", repr(x))

    key = key_func if key_func is not None else default_key

    index = {}
    for i, x in enumerate(elts):
        k = key(x)

        if k in index:
            old = elts[index[k] - 1]
            if old != x:
                raise ValueError(
                    "Key-Kollision: zwei verschiedene Gruppenelemente haben denselben Key. "
                    "Bitte gib eine eigene key_func an."
                )

        index[k] = i + 1

    def action_perm(a):
        """
        Permutation der Menge G durch Multiplikation mit a.
        """
        images = []

        for x in elts:
            if side == "left":
                y = a * x
            else:
                y = x * a

            ky = key(y)

            if ky not in index:
                raise ValueError(
                    "Produkt liegt nicht in der Index-Tabelle. "
                    "Eventuell ist key_func nicht stabil genug."
                )

            images.append(index[ky])

        return S(images)

    gens = list(G.gens())

    if len(gens) == 0:
        # triviale Gruppe
        pgens = [S([1])]
    else:
        pgens = [action_perm(a) for a in gens]

    P = PermutationGroup(pgens)

    if check:
        if P.order() != G.order():
            raise ValueError(
                "Die Permutationsgruppe hat nicht dieselbe Ordnung wie G: "
                f"{P.order()} statt {G.order()}."
            )

    if not return_data:
        return P

    return {
        "group": P,
        "elements": elts,
        "index": index,
        "perm_of_gen": dict(zip(gens, pgens)),
        "S": S,
        "key": key,
    }

# ------------------------------------------------------------
# Einstellungen
# ------------------------------------------------------------

# Beispiele:
# G = SymmetricGroup(4)
# G = SymmetricGroup(5)
# G = DihedralGroup(8)
# G = AlternatingGroup(5)
# G = CyclicPermutationGroup(12)
# G = AbelianGroup([4, 4])

#shuffle(chords)

#G = CyclicPermutationGroup(37) #AbelianGroup([2,3,5])

G = regular_permutation_group(MatrixGroup([
    matrix(ZZ, 4, 4, [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0,-1, 0,
        0, 0, 0,-1
    ]),
    matrix(ZZ, 4, 4, [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0,-1,-1,
        0, 0, 1, 0
    ]),
    matrix(ZZ, 4, 4, [
       -1, 1, 0, 0,
       -1, 0, 0, 0,
        0, 0, 0, 1,
        0, 0,-1,-1
    ]),
    matrix(ZZ, 4, 4, [
        1,-1, 0, 0,
        0,-1, 0, 0,
        0, 0, 0, 1,
        0, 0, 1, 0
    ])
]))["group"]

G = regular_permutation_group(DihedralGroup(4))["group"]

for g in G.gens():
    print(Permutation(g))
print("Order:", G.order())

#G = regular_permutation_group(MatrixGroup([ 
#                  matrix([[0,0,0,1],[0,0,-1,0],[0,1,0,0],[-1,0,0,0]]),
#                  matrix([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]]),
#                  matrix([[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]]),
#                  matrix([[0,1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,-1,0]])]) )["group"]

#tempo = 80
#voices = 3
#base_duration = 1/7
#chord_change_every = G.order()

#G = AbelianGroup([2, 9, 5])
tempo = 80
voices = 5
base_duration=1/7
chord_change_every = 3
pi_nr = exp(-pi)
pi_digits = 10000
octave_range = (-2, 2)
add21 = False

iinds = group_music(
    G,
    chords,
    tempo=tempo,
    base_duration=base_duration,
    octave_range=octave_range,
    pi_nr = pi_nr,
    pi_digits=pi_digits,
    voices=voices,
    chord_change_every=chord_change_every,
    iter_it = 2
)

fn = "./group_D4_iter_2.mid"

writePitches(
    fn,
    iinds,
    tempo=tempo,
    instrument=len(iinds) * [0],
    add21=False,
    start_at=len(iinds) * [0],
    durationsInQuarterNotes=False
)

groups_ = [
        #(AlternatingGroup(4), "A4"),
        (SymmetricGroup(2), "C2"),
        (CyclicPermutationGroup(3), "C3"),
        (CyclicPermutationGroup(4), "C4"),
        (KleinFourGroup(), "KleinFourGroup"),
        (CyclicPermutationGroup(5), "C5"),
        (CyclicPermutationGroup(6), "C6"),
        (SymmetricGroup(3), "S3"),
        (CyclicPermutationGroup(7), "C7"),
        (CyclicPermutationGroup(8), "C8"),
        (direct_product_permgroups([CyclicPermutationGroup(4), CyclicPermutationGroup(2)]), "C2xC4"),
        (direct_product_permgroups([CyclicPermutationGroup(2), CyclicPermutationGroup(2), CyclicPermutationGroup(2)]), "C2xC2xC2"),
        (DihedralGroup(4), "D4"),
        (QuaternionGroup(), "Quaternions"),
        (CyclicPermutationGroup(9), "C9"),
        (direct_product_permgroups([CyclicPermutationGroup(3), CyclicPermutationGroup(3)]), "C3xC3"),
        (CyclicPermutationGroup(10), "C10"),
        (DihedralGroup(5), "D5"),
        (CyclicPermutationGroup(11), "C11"),
        (CyclicPermutationGroup(12), "C12"),
        (direct_product_permgroups([CyclicPermutationGroup(6), CyclicPermutationGroup(2)]), "C6xC2"),
        (DihedralGroup(6), "D6"),
        (AlternatingGroup(4), "A4"),
        (DiCyclicGroup(3), "C3_C4"),
        (CyclicPermutationGroup(13), "C13"),
        (CyclicPermutationGroup(14), "C14"),
        (DihedralGroup(7), "D7"),
        (CyclicPermutationGroup(15), "C15"),
        (CyclicPermutationGroup(16), "C16"),
        (DihedralGroup(8),"D8"),
        (PermutationGroup([[(1,2,3,4,5,6,7,8),(9,10,11,12,13,14,15,16)], [(1,12,5,16),(2,11,6,15),(3,10,7,14),(4,9,8,13)]]), "Q16"),
        (PermutationGroup([[(1,2,3,4,5,6,7,8)], [(2,4),(3,7),(6,8)]]),"SD16"),
        (PermutationGroup([[(1,2,3,4,5,6,7,8)], [(2,6),(4,8)]]),"M4_2"),
        (PermutationGroup([[(1,2,3,4),(5,6,7,8)], [(1,4,3,2),(5,6,7,8)], [(1,6),(2,7),(3,8),(4,5)]]),"C4oD4"),
        (direct_product_permgroups([CyclicPermutationGroup(8), CyclicPermutationGroup(2)]), "C8xC2"),
        (CyclicPermutationGroup(17),"C17"),
        
        (DihedralGroup(9), "D9"),
        (DihedralGroup(10), "D10"),
        (DihedralGroup(11), "D11"),
        (DihedralGroup(12), "D12"),
        (SymmetricGroup(4), "S4"),
        (direct_product_permgroups([CyclicPermutationGroup(3), CyclicPermutationGroup(3),CyclicPermutationGroup(3)]), "C3xC3xC3"),
    ]


for group,name in groups_:
    print(name)
    G = regular_permutation_group(group)["group"]

    for g in G.gens():
        print(Permutation(g))

    
