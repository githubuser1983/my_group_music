# -*- coding: utf-8 -*-
"""
group_midi_permuter.py

Pure-Python port of group_midi_permuter.sage.

Apply permutations from a finite group sequentially to blocks of MIDI onset
positions.  Uses mido only for MIDI I/O; no SageMath, MIDIUtil or MidiFile.py.

Install:
    pip install mido

Example:
    python group_midi_permuter.py fuer_elise.mid fuer_elise_D4_python.mid --group D4

The default is the regular left action.  D4 has order 8, therefore it acts on
blocks of 8 distinct onset positions.  Simultaneous notes move together.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Sequence

from group_music import (
    FiniteGroup,
    Permutation,
    group_action_permutations,
    parse_group,
)


def _require_mido():
    try:
        import mido
        return mido
    except ImportError as exc:
        raise ImportError("Install mido with: pip install mido") from exc


# =============================================================================
# MIDI reading
# =============================================================================


def read_midi_structure(filename: str) -> dict[str, Any]:
    """
    Read notes plus all non-note events.

    Each note keeps source track, channel, pitch, velocity, absolute onset tick,
    duration in ticks and source ordering.  Tempo, time signature, program,
    control/pedal and other non-note messages keep their original absolute time.
    """
    mido = _require_mido()
    mid = mido.MidiFile(filename)

    notes = []
    non_note_events = [[] for _ in mid.tracks]
    unmatched_off = 0
    unmatched_on = 0

    for track_index, track in enumerate(mid.tracks):
        abs_tick = 0
        active = defaultdict(deque)

        for message_order, msg in enumerate(track):
            abs_tick += int(msg.time)
            is_note_on = msg.type == "note_on" and msg.velocity > 0
            is_note_off = msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)

            if is_note_on:
                key = (int(msg.channel), int(msg.note))
                active[key].append({"on": abs_tick, "velocity": int(msg.velocity), "order": message_order})
                continue

            if is_note_off:
                key = (int(msg.channel), int(msg.note))
                if active[key]:
                    start = active[key].popleft()
                    notes.append({
                        "track": track_index,
                        "channel": int(msg.channel),
                        "note": int(msg.note),
                        "velocity": int(start["velocity"]),
                        "on": int(start["on"]),
                        "duration": max(1, int(abs_tick - start["on"])),
                        "source_order": int(start["order"]),
                    })
                else:
                    unmatched_off += 1
                continue

            if msg.type != "end_of_track":
                non_note_events[track_index].append((abs_tick, message_order, msg.copy(time=0)))

        for q in active.values():
            unmatched_on += len(q)

    return {
        "mid": mid,
        "notes": notes,
        "non_note_events": non_note_events,
        "unmatched_note_on": unmatched_on,
        "unmatched_note_off": unmatched_off,
    }


# =============================================================================
# Permute onset blocks
# =============================================================================


def _as_vector(p: Permutation | Sequence[int]) -> list[int]:
    if isinstance(p, Permutation):
        return list(p.images)
    return [int(x) for x in p]


def _apply_permutations_to_note_subset(
    notes: Sequence[dict[str, Any]],
    permutations: Sequence[Permutation | Sequence[int]],
    block_size: int,
    partial_block: str = "keep",
    inverse: bool = False,
):
    if partial_block not in ("keep", "drop"):
        raise ValueError("partial_block must be 'keep' or 'drop'")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not permutations:
        raise ValueError("no permutations supplied")

    out = [dict(note) for note in notes]
    by_tick = defaultdict(list)
    for idx, note in enumerate(out):
        by_tick[int(note["on"])].append(idx)

    onset_ticks = sorted(by_tick)
    complete_blocks = len(onset_ticks) // block_size
    dropped_indices = set()

    for block_nr in range(complete_blocks):
        start = block_nr * block_size
        block_ticks = onset_ticks[start : start + block_size]
        p = _as_vector(permutations[block_nr % len(permutations)])
        if len(p) != block_size or sorted(p) != list(range(block_size)):
            raise ValueError("permutation size/content does not match block size")

        if inverse:
            q = [0] * block_size
            for source, target in enumerate(p):
                q[target] = source
            p = q

        for source_slot, source_tick in enumerate(block_ticks):
            target_tick = block_ticks[p[source_slot]]
            for note_index in by_tick[source_tick]:
                out[note_index]["on"] = int(target_tick)

    remainder_start = complete_blocks * block_size
    if remainder_start < len(onset_ticks) and partial_block == "drop":
        for tick in onset_ticks[remainder_start:]:
            dropped_indices.update(by_tick[tick])

    if dropped_indices:
        out = [note for i, note in enumerate(out) if i not in dropped_indices]

    return out, complete_blocks, len(onset_ticks) - complete_blocks * block_size


def permute_midi_notes(
    notes: Sequence[dict[str, Any]],
    permutations: Sequence[Permutation | Sequence[int]],
    block_size: int,
    scope: str = "global",
    partial_block: str = "keep",
    inverse: bool = False,
):
    """
    scope='global': one onset grid across all tracks.
    scope='track': each track is permuted independently.
    """
    if scope == "global":
        transformed, blocks, remainder = _apply_permutations_to_note_subset(
            notes, permutations, block_size, partial_block=partial_block, inverse=inverse
        )
        return transformed, {"global": (blocks, remainder)}

    if scope == "track":
        by_track = defaultdict(list)
        for note in notes:
            by_track[note["track"]].append(note)
        transformed = []
        stats = {}
        for track_index in sorted(by_track):
            part, blocks, remainder = _apply_permutations_to_note_subset(
                by_track[track_index], permutations, block_size,
                partial_block=partial_block, inverse=inverse
            )
            transformed.extend(part)
            stats[track_index] = (blocks, remainder)
        return transformed, stats

    raise ValueError("scope must be 'global' or 'track'")


# =============================================================================
# MIDI writing
# =============================================================================


def _event_priority(msg) -> int:
    if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
        return 20
    if msg.type == "note_on":
        return 30
    if msg.type in ("program_change", "control_change", "pitchwheel"):
        return 10
    if getattr(msg, "is_meta", False):
        return 0
    return 15


def write_midi_structure(parsed: dict[str, Any], transformed_notes, output_filename: str) -> str:
    mido = _require_mido()
    original = parsed["mid"]
    out_mid = mido.MidiFile(type=original.type, ticks_per_beat=original.ticks_per_beat)

    notes_by_track = defaultdict(list)
    for note in transformed_notes:
        notes_by_track[note["track"]].append(note)

    for track_index in range(len(original.tracks)):
        out_track = mido.MidiTrack()
        events = []
        serial = 0

        for abs_tick, original_order, msg in parsed["non_note_events"][track_index]:
            events.append((int(abs_tick), _event_priority(msg), int(original_order), serial, msg.copy(time=0)))
            serial += 1

        for note in notes_by_track.get(track_index, []):
            on_tick = int(note["on"])
            off_tick = int(note["on"] + note["duration"])
            on_msg = mido.Message(
                "note_on", channel=int(note["channel"]), note=int(note["note"]),
                velocity=int(note["velocity"]), time=0
            )
            off_msg = mido.Message(
                "note_off", channel=int(note["channel"]), note=int(note["note"]),
                velocity=0, time=0
            )
            source_order = int(note.get("source_order", 0))
            events.append((on_tick, 30, source_order, serial, on_msg)); serial += 1
            events.append((off_tick, 20, source_order, serial, off_msg)); serial += 1

        events.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        last_tick = 0
        for abs_tick, _priority, _original_order, _serial, msg in events:
            delta = max(0, int(abs_tick) - last_tick)
            out_track.append(msg.copy(time=delta))
            last_tick = int(abs_tick)
        out_track.append(mido.MetaMessage("end_of_track", time=0))
        out_mid.tracks.append(out_track)

    out_mid.save(output_filename)
    return output_filename


# =============================================================================
# High-level API
# =============================================================================


def apply_group_to_midi(
    input_filename: str,
    output_filename: str,
    G: FiniteGroup,
    representation: str = "regular",
    side: str = "left",
    scope: str = "global",
    identity_first: bool = True,
    partial_block: str = "keep",
    inverse: bool = False,
    verbose: bool = True,
):
    action = group_action_permutations(
        G, representation=representation, side=side, identity_first=identity_first
    )
    parsed = read_midi_structure(input_filename)
    transformed_notes, stats = permute_midi_notes(
        parsed["notes"], action["permutations"], action["degree"],
        scope=scope, partial_block=partial_block, inverse=inverse
    )
    write_midi_structure(parsed, transformed_notes, output_filename)

    if verbose:
        print("Input MIDI:       ", input_filename)
        print("Output MIDI:      ", output_filename)
        print("Group order:      ", len(action["elements"]))
        print("Representation:   ", action["representation"])
        if action["representation"] == "regular":
            print("Action side:      ", action["side"])
        print("Permutation size: ", action["degree"])
        print("Scope:            ", scope)
        print("Notes read:       ", len(parsed["notes"]))
        print("Notes written:    ", len(transformed_notes))
        print("Block stats:      ", stats)
        if parsed["unmatched_note_on"] or parsed["unmatched_note_off"]:
            print("Warning: unmatched note messages:", parsed["unmatched_note_on"], parsed["unmatched_note_off"])
        print("Non-note MIDI events remain at their original absolute times.")

    return {
        "output": output_filename,
        "action": action,
        "stats": stats,
        "notes_in": len(parsed["notes"]),
        "notes_out": len(transformed_notes),
    }


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply finite-group permutations to a MIDI file.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--group", default="D4", help="D4, C8, S4, A4, V4, Q8, abelian:2,9,5")
    parser.add_argument("--representation", choices=["regular", "given"], default="regular")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--scope", choices=["global", "track"], default="global")
    parser.add_argument("--partial-block", choices=["keep", "drop"], default="keep")
    parser.add_argument("--inverse", action="store_true")
    args = parser.parse_args()

    G = parse_group(args.group)
    apply_group_to_midi(
        args.input, args.output, G,
        representation=args.representation,
        side=args.side,
        scope=args.scope,
        identity_first=True,
        partial_block=args.partial_block,
        inverse=args.inverse,
        verbose=True,
    )


if __name__ == "__main__":
    _main()
