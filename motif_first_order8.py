#!/usr/bin/env python3
"""Motif-first music for all five isomorphism classes of groups of order 8.

A controlled comparison with d4_motif_composer.py: the harmony, melody template,
three-track orchestration, note lengths, tempo, articulation, and phrase count
stay fixed. Only the left-regular permutation chosen for a phrase changes.

Dependencies: pip install mido; place group_music.py and d4_motif_composer.py
next to this file. No SageMath and no MIDIUtil.

Examples:
  python motif_first_order8.py --group all --output-dir rendered
  python motif_first_order8.py --group Q8 --cycles 4 --tempo 80
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import mido

import d4_motif_composer as motif
from group_music import AbelianGroup, DihedralGroup, QuaternionGroup, regular_action_permutations

GROUP_NAMES = ("C8", "C4xC2", "C2xC2xC2", "D4", "Q8")


def make_group(name):
    if name == "C8":
        return AbelianGroup([8])
    if name == "C4xC2":
        return AbelianGroup([4, 2])
    if name == "C2xC2xC2":
        return AbelianGroup([2, 2, 2])
    if name == "D4":
        return DihedralGroup(4)  # the square's dihedral group; order 8
    if name == "Q8":
        return QuaternionGroup()
    raise ValueError(f"Unsupported order-eight group: {name}")


def group_element_order(element, identity):
    value = identity
    for k in range(1, 9):
        value = value * element
        if value == identity:
            return k
    raise AssertionError("element order did not divide eight")


def validated_action(group):
    action = regular_action_permutations(group, side="left", identity_first=True)
    perms = action["permutations"]
    assert group.order() == len(action["elements"]) == len(perms) == 8
    assert len({p.images for p in perms}) == 8
    assert all(p.degree == 8 for p in perms)
    assert perms[0].is_identity()
    # Faithfulness and the homomorphism law L_a * L_b = L_(ab).
    ids = {g: i for i, g in enumerate(action["elements"])}
    for a in action["elements"]:
        for b in action["elements"]:
            assert perms[ids[a * b]] == perms[ids[a]] * perms[ids[b]]
    return action


def make_piece(group_name="Q8", tempo=80, cycles=4):
    if not 30 <= tempo <= 200:
        raise ValueError("tempo must be between 30 and 200")
    if not 1 <= cycles <= 20:
        raise ValueError("cycles must be between 1 and 20")
    group = make_group(group_name)
    action = validated_action(group)
    perms = action["permutations"]
    voices = [[], [], []]  # melody, accompaniment, bass
    structure = []

    for cycle in range(cycles):
        for position, chord in enumerate(motif.PROGRESSION):
            phrase_no = cycle * len(motif.PROGRESSION) + position
            phrase_start = phrase_no * motif.PHRASE_BEATS
            group_index = (position + 3 * cycle) % len(perms)
            p = perms[group_index]
            structure.append({
                "phrase": phrase_no + 1,
                "harmony": chord.name,
                "group_element": repr(action["elements"][group_index]),
                "permutation_zero_based": list(p.images),
            })

            # Same recognizable melodic anchors and restrained neighboring degrees.
            tonic = chord.root + 12
            dynamic = (74, 79, 82, 71)[cycle % 4]
            for i, (degree, start, span) in enumerate(zip(
                motif.MOTIF_DEGREES, motif.MOTIF_STARTS, motif.MOTIF_SPANS
            )):
                d = motif.motif_variation(degree, p.images[i], i)
                pitch = motif.scale_pitch(tonic, chord.mode, d)
                gate = 0.82 if i != 7 else 0.72
                accent = 4 if i in (0, 3, 6) else 0
                voices[0].append(motif.NoteEvent(
                    phrase_start + start,
                    phrase_start + start + span * gate,
                    pitch,
                    dynamic + accent - (3 if i in (1, 4) else 0),
                ))

            # This is the exact regular action on the eight accompaniment tokens.
            acc_root = chord.root + (12 if chord.root < 48 else 0)
            chord_pitches = tuple(acc_root + d for d in motif.CHORD_OFFSETS[chord.mode])
            for j in range(8):
                token = p.images[j]
                voices[1].append(motif.NoteEvent(
                    phrase_start + j,
                    phrase_start + j + motif.ACCOMP_GATES[token],
                    chord_pitches[motif.ACCOMP_DEGREES[token]],
                    motif.ACCOMP_VELOCITIES[token] + (cycle % 3) * 2,
                ))

            bass_root = chord.root - 12 if chord.root >= 46 else chord.root
            bass_velocity = (55, 57, 59, 52)[cycle % 4]
            voices[2].append(motif.NoteEvent(
                phrase_start, phrase_start + 3.15, bass_root, bass_velocity
            ))
            voices[2].append(motif.NoteEvent(
                phrase_start + 4, phrase_start + 7.05,
                bass_root + (7 if chord.name == "A7" else 0), bass_velocity - 5,
            ))

    end = len(motif.PROGRESSION) * cycles * motif.PHRASE_BEATS
    voices[0].append(motif.NoteEvent(end, end + 3.5, 62, 72))
    voices[1].extend(motif.NoteEvent(end, end + 3.2, p, 40) for p in (53, 57, 60))
    voices[2].append(motif.NoteEvent(end, end + 3.4, 38, 54))
    return voices, structure, end + 4.0


def write_midi(path: Path, voices, total_beats, tempo, group_name):
    """Keep D4's earlier MIDI byte-identical; label the other groups accurately."""
    mid = mido.MidiFile(type=1, ticks_per_beat=motif.TICKS_PER_BEAT)
    labels = (
        "Melody - stable eight-note motif",
        f"{group_name}-transformed chord arpeggio",
        "Bass",
    )
    for channel, (label, notes) in enumerate(zip(labels, voices)):
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=label, time=0))
        events = []
        if channel == 0:
            events.extend([
                (0, 0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0)),
                (0, 0, mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0)),
                (0, 0, mido.MetaMessage("key_signature", key="Dm", time=0)),
            ])
        events.append((0, 1, mido.Message("program_change", channel=channel, program=0, time=0)))
        for note in notes:
            on = round(note.start_beats * motif.TICKS_PER_BEAT)
            off = max(on + 1, round(note.end_beats * motif.TICKS_PER_BEAT))
            events.append((on, 3, mido.Message(
                "note_on", note=note.pitch, velocity=note.velocity, channel=channel, time=0
            )))
            events.append((off, 2, mido.Message(
                "note_off", note=note.pitch, velocity=0, channel=channel, time=0
            )))
        events.sort(key=lambda x: (x[0], x[1]))
        previous = 0
        for tick, _, msg in events:
            track.append(msg.copy(time=tick - previous))
            previous = tick
        final_tick = round(total_beats * motif.TICKS_PER_BEAT)
        track.append(mido.MetaMessage("end_of_track", time=final_tick - previous))
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))
    return path


def generate(group_name, output_dir, tempo=80, cycles=4):
    G = make_group(group_name)
    voices, structure, end = make_piece(group_name, tempo, cycles)
    midi_file = output_dir / f"motif_first_{group_name}.mid"
    write_midi(midi_file, voices, end, tempo, group_name)
    element_orders = dict(sorted(Counter(
        group_element_order(x, G.identity()) for x in G
    ).items()))
    report = {
        "group": group_name,
        "order": G.order(),
        "element_order_distribution": element_orders,
        "tempo_bpm": tempo,
        "cycles": cycles,
        "number_of_phrases": len(structure),
        "midi_note_count": sum(map(len, voices)),
        "notes_per_voice": [len(v) for v in voices],
        "duration_beats": end,
        "duration_seconds": end * 60 / tempo,
        "group_action": "left-regular, degree eight",
        "phrases": structure,
    }
    (output_dir / f"motif_first_{group_name}_structure.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return midi_file, report, voices


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", choices=("all",) + GROUP_NAMES, default="all")
    p.add_argument("--output-dir", type=Path, default=Path("motif_order8_outputs"))
    p.add_argument("--tempo", type=int, default=80)
    p.add_argument("--cycles", type=int, default=4)
    args = p.parse_args()
    names = GROUP_NAMES if args.group == "all" else (args.group,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        midi, report, _ = generate(name, args.output_dir, args.tempo, args.cycles)
        print(f"{name:9s} |G|={report['order']} order profile={report['element_order_distribution']} "
              f"notes={report['midi_note_count']} duration={report['duration_seconds']:.1f}s MIDI={midi}")


if __name__ == "__main__":
    main()
