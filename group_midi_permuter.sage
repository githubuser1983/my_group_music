# -*- coding: utf-8 -*-
"""
group_midi_permuter.sage

Apply permutations coming from a finite SageMath group to an existing MIDI file.

Main musical idea
-----------------
Treat consecutive MIDI onset positions as slots.  If the chosen group action has
n slots, split the melody into consecutive blocks of n onset positions.  On
block 0 apply the permutation belonging to the first group element, on block 1
the second one, and so on.  After all group elements have been used, cycle.

By default the REGULAR LEFT ACTION is used:

    L_g : G -> G,  x |-> g*x.

Thus every g in G determines a permutation of |G| positions.  The source onset
slot i is moved to target slot sigma_g(i).  Pitch, velocity, channel, track and
note duration are preserved; only onset positions are permuted.

This is especially natural for a monophonic melody such as fuer_elise.mid, but
simultaneous notes are kept together as one onset slice, so chords also work.

Dependency
----------
The script uses mido for MIDI I/O.  In a Sage environment install it once with:

    sage -pip install mido

Example
-------
    G = DihedralGroup(4)    # order 8 -> blocks of 8 onset positions

    apply_group_to_midi(
        "fuer_elise.mid",
        "fuer_elise_D4.mid",
        G,
        representation="regular",
        side="left",
        scope="global",
        identity_first=True,
        partial_block="keep",
        verbose=True,
    )

Alternative representation
--------------------------
If G is already a permutation group, representation="given" uses the natural
permutation action of its elements instead of the regular action.  For example,
SymmetricGroup(4) then acts on blocks of 4 onset positions, whereas its regular
action would act on blocks of 24 positions.
"""

from collections import defaultdict, deque


# -----------------------------------------------------------------------------
# Sage group -> permutation vectors
# -----------------------------------------------------------------------------

def _default_key(x):
    """Robust key for many Sage group elements."""
    try:
        hash(x)
        return ("object", x)
    except TypeError:
        pass

    if hasattr(x, "matrix"):
        try:
            M = x.matrix()
            return ("matrix", tuple(M.list()))
        except Exception:
            pass

    try:
        t = tuple(x)
        hash(t)
        return ("tuple", t)
    except Exception:
        pass

    return ("repr", repr(x))


def _ordered_group_elements(G, identity_first=True):
    """
    Return a deterministic Sage enumeration, optionally moving the identity
    element to the first position without changing the relative order otherwise.
    """
    elts = list(G)

    if not identity_first or len(elts) <= 1:
        return elts

    try:
        one = G.one()
    except Exception:
        try:
            one = G.identity()
        except Exception:
            return elts

    for i, g in enumerate(elts):
        if g == one:
            if i != 0:
                elts = [elts[i]] + elts[:i] + elts[i+1:]
            break

    return elts


def regular_action_permutations(G, side="left", identity_first=True, key_func=None):
    r"""
    Return all permutations of the regular action of the finite group G.

    Output dictionary:
        elements       ordered group elements
        permutations   list of 0-based permutation vectors
        degree         |G|

    A vector p means
        source slot i  --->  target slot p[i].

    For side="left":
        p_g encodes x |-> g*x.

    For side="right":
        p_g encodes x |-> x*g.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")

    elts = _ordered_group_elements(G, identity_first=identity_first)
    n = len(elts)

    if n == 0:
        raise ValueError("The group appears to have no elements.")

    key = key_func if key_func is not None else _default_key

    index = {}
    for i, x in enumerate(elts):
        k = key(x)
        if k in index:
            old = elts[index[k]]
            if old != x:
                raise ValueError(
                    "Key collision for group elements. Supply a custom key_func."
                )
        index[k] = i

    perms = []

    for g in elts:
        p = []
        for x in elts:
            y = g * x if side == "left" else x * g
            ky = key(y)
            if ky not in index:
                raise ValueError(
                    "A group product was not found in the element index. "
                    "Supply a more stable key_func."
                )
            p.append(index[ky])

        if sorted(p) != list(range(n)):
            raise ValueError("The regular action did not produce a permutation.")

        perms.append(p)

    return {
        "elements": elts,
        "permutations": perms,
        "degree": n,
        "representation": "regular",
        "side": side,
    }


def given_action_permutations(G, identity_first=True):
    """
    Use the natural action when G is already a Sage permutation group.

    If the natural degree is d, each group element is converted to a vector
    [g(1)-1, ..., g(d)-1].
    """
    elts = _ordered_group_elements(G, identity_first=identity_first)

    try:
        d = int(G.degree())
    except Exception as exc:
        raise TypeError(
            "representation='given' requires a Sage permutation group "
            "with a degree() method."
        ) from exc

    if d <= 0:
        raise ValueError("Permutation degree must be positive.")

    perms = []
    for g in elts:
        try:
            p = [int(g(i + 1)) - 1 for i in range(d)]
        except Exception as exc:
            raise TypeError(
                "Could not evaluate a group element as a permutation."
            ) from exc

        if sorted(p) != list(range(d)):
            raise ValueError("A group element did not define a permutation of 1..degree.")

        perms.append(p)

    return {
        "elements": elts,
        "permutations": perms,
        "degree": d,
        "representation": "given",
        "side": None,
    }


def group_action_permutations(
    G,
    representation="regular",
    side="left",
    identity_first=True,
    key_func=None,
):
    """Dispatch to the requested group representation."""
    if representation == "regular":
        return regular_action_permutations(
            G,
            side=side,
            identity_first=identity_first,
            key_func=key_func,
        )

    if representation == "given":
        return given_action_permutations(
            G,
            identity_first=identity_first,
        )

    raise ValueError("representation must be 'regular' or 'given'.")


# -----------------------------------------------------------------------------
# MIDI reading
# -----------------------------------------------------------------------------

def _import_mido():
    try:
        import mido
        return mido
    except ImportError as exc:
        raise ImportError(
            "This script needs the Python package 'mido'. Install it in Sage with:\n"
            "    sage -pip install mido"
        ) from exc


def read_midi_structure(filename):
    """
    Read a MIDI file and return notes plus all non-note messages.

    Every note is represented by a dictionary containing absolute onset tick,
    duration in ticks, pitch, velocity, channel and source track.

    Non-note messages keep their original absolute time.  This means tempo,
    time signatures, program changes and controllers are preserved at their
    original positions.  In particular, sustain-pedal data is NOT permuted.
    """
    mido = _import_mido()
    mid = mido.MidiFile(filename)

    notes = []
    non_note_events = [[] for _ in mid.tracks]
    unmatched_off = 0
    unmatched_on = 0

    # Per track, because a MIDI note-off belongs to the channel stream in that track.
    for track_index, track in enumerate(mid.tracks):
        abs_tick = 0
        active = defaultdict(deque)

        for message_order, msg in enumerate(track):
            abs_tick += int(msg.time)

            is_note_on = (msg.type == "note_on" and msg.velocity > 0)
            is_note_off = (
                msg.type == "note_off"
                or (msg.type == "note_on" and msg.velocity == 0)
            )

            if is_note_on:
                key = (int(msg.channel), int(msg.note))
                active[key].append({
                    "on": abs_tick,
                    "velocity": int(msg.velocity),
                    "order": message_order,
                })
                continue

            if is_note_off:
                key = (int(msg.channel), int(msg.note))
                if active[key]:
                    start = active[key].popleft()
                    duration = max(1, abs_tick - start["on"])
                    notes.append({
                        "track": track_index,
                        "channel": int(msg.channel),
                        "note": int(msg.note),
                        "velocity": int(start["velocity"]),
                        "on": int(start["on"]),
                        "duration": int(duration),
                        "source_order": int(start["order"]),
                    })
                else:
                    unmatched_off += 1
                continue

            if msg.type != "end_of_track":
                non_note_events[track_index].append(
                    (abs_tick, message_order, msg.copy(time=0))
                )

        for queue in active.values():
            unmatched_on += len(queue)

    return {
        "mid": mid,
        "notes": notes,
        "non_note_events": non_note_events,
        "unmatched_note_on": unmatched_on,
        "unmatched_note_off": unmatched_off,
    }


# -----------------------------------------------------------------------------
# Apply a sequence of permutations to onset blocks
# -----------------------------------------------------------------------------

def _apply_permutations_to_note_subset(
    notes,
    permutations,
    block_size,
    partial_block="keep",
    inverse=False,
):
    """
    Transform one note subset (globally or one track).

    Distinct onset ticks are the slots.  Simultaneous notes form one onset slice
    and always move together.

    For a direct permutation p:
        source slot i -> target slot p[i].

    If inverse=True, the inverse vector is used instead.
    """
    if partial_block not in ("keep", "drop"):
        raise ValueError("partial_block must be 'keep' or 'drop'.")

    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    if not permutations:
        raise ValueError("No permutations were supplied.")

    # Copy so that the caller's parsed data remains unchanged.
    out = [dict(note) for note in notes]

    by_tick = defaultdict(list)
    for idx, note in enumerate(out):
        by_tick[note["on"]].append(idx)

    onset_ticks = sorted(by_tick)
    complete_blocks = len(onset_ticks) // block_size

    dropped_indices = set()

    for block_nr in range(complete_blocks):
        start = block_nr * block_size
        block_ticks = onset_ticks[start:start + block_size]
        p = list(permutations[block_nr % len(permutations)])

        if len(p) != block_size:
            raise ValueError("Permutation size does not match block size.")

        if inverse:
            q = [None] * block_size
            for i, target in enumerate(p):
                q[target] = i
            p = q

        # The rhythmic grid stays fixed; onset slices move among its slots.
        for source_slot, source_tick in enumerate(block_ticks):
            target_slot = p[source_slot]
            target_tick = block_ticks[target_slot]

            for note_index in by_tick[source_tick]:
                note = out[note_index]
                note["on"] = int(target_tick)

    remainder_start = complete_blocks * block_size
    if remainder_start < len(onset_ticks) and partial_block == "drop":
        for tick in onset_ticks[remainder_start:]:
            dropped_indices.update(by_tick[tick])

    if dropped_indices:
        out = [note for i, note in enumerate(out) if i not in dropped_indices]

    return out, complete_blocks, len(onset_ticks) - complete_blocks * block_size


def permute_midi_notes(
    notes,
    permutations,
    block_size,
    scope="global",
    partial_block="keep",
    inverse=False,
):
    """
    Apply the permutation schedule either globally or independently per track.

    scope="global":
        all tracks share one global onset grid; simultaneous cross-track notes
        move together.

    scope="track":
        each track gets its own onset grid and starts the group sequence at the
        identity independently.
    """
    if scope == "global":
        transformed, blocks, remainder = _apply_permutations_to_note_subset(
            notes,
            permutations,
            block_size,
            partial_block=partial_block,
            inverse=inverse,
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
                by_track[track_index],
                permutations,
                block_size,
                partial_block=partial_block,
                inverse=inverse,
            )
            transformed.extend(part)
            stats[track_index] = (blocks, remainder)

        return transformed, stats

    raise ValueError("scope must be 'global' or 'track'.")


# -----------------------------------------------------------------------------
# MIDI writing
# -----------------------------------------------------------------------------

def _event_priority(msg):
    """Ordering for messages that land on exactly the same absolute tick."""
    if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
        return 20
    if msg.type == "note_on":
        return 30
    if msg.type in ("program_change", "control_change", "pitchwheel"):
        return 10
    if getattr(msg, "is_meta", False):
        return 0
    return 15


def write_midi_structure(parsed, transformed_notes, output_filename):
    """
    Write transformed notes while preserving non-note events at original times.
    """
    mido = _import_mido()
    original = parsed["mid"]

    out_mid = mido.MidiFile(
        type=original.type,
        ticks_per_beat=original.ticks_per_beat,
        charset=getattr(original, "charset", "latin1"),
        clip=getattr(original, "clip", False),
    )

    notes_by_track = defaultdict(list)
    for note in transformed_notes:
        notes_by_track[note["track"]].append(note)

    for track_index in range(len(original.tracks)):
        out_track = mido.MidiTrack()
        events = []
        serial = 0

        # Preserve meta/controller/program/etc. events at original absolute ticks.
        for abs_tick, original_order, msg in parsed["non_note_events"][track_index]:
            events.append((
                int(abs_tick),
                _event_priority(msg),
                int(original_order),
                serial,
                msg.copy(time=0),
            ))
            serial += 1

        # Insert the transformed notes.
        for note in notes_by_track.get(track_index, []):
            on_tick = int(note["on"])
            off_tick = int(note["on"] + note["duration"])

            on_msg = mido.Message(
                "note_on",
                channel=int(note["channel"]),
                note=int(note["note"]),
                velocity=int(note["velocity"]),
                time=0,
            )
            off_msg = mido.Message(
                "note_off",
                channel=int(note["channel"]),
                note=int(note["note"]),
                velocity=0,
                time=0,
            )

            source_order = int(note.get("source_order", 0))
            events.append((on_tick, 30, source_order, serial, on_msg))
            serial += 1
            events.append((off_tick, 20, source_order, serial, off_msg))
            serial += 1

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


# -----------------------------------------------------------------------------
# Public high-level function
# -----------------------------------------------------------------------------

def apply_group_to_midi(
    input_filename,
    output_filename,
    G,
    representation="regular",
    side="left",
    scope="global",
    identity_first=True,
    partial_block="keep",
    inverse=False,
    key_func=None,
    verbose=True,
):
    r"""
    Read input_filename, apply group permutations sequentially to onset blocks,
    and write output_filename.

    The sequence is:
        block 0 -> permutation of group element g_0
        block 1 -> permutation of group element g_1
        ...
        block k -> permutation of g_(k mod |G|)

    For representation="regular", the block size is |G|.
    For representation="given", the block size is G.degree().

    Notes keep pitch, velocity, duration, channel and track.  Only onset slots
    are permuted.  The rhythmic grid itself therefore remains the same.
    """
    action = group_action_permutations(
        G,
        representation=representation,
        side=side,
        identity_first=identity_first,
        key_func=key_func,
    )

    parsed = read_midi_structure(input_filename)

    transformed_notes, stats = permute_midi_notes(
        parsed["notes"],
        action["permutations"],
        action["degree"],
        scope=scope,
        partial_block=partial_block,
        inverse=inverse,
    )

    write_midi_structure(parsed, transformed_notes, output_filename)

    if verbose:
        print("Input MIDI:      ", input_filename)
        print("Output MIDI:     ", output_filename)
        print("Group order:     ", len(action["elements"]))
        print("Representation:  ", action["representation"])
        if action["representation"] == "regular":
            print("Action side:     ", action["side"])
        print("Permutation size:", action["degree"])
        print("Scope:           ", scope)
        print("Notes read:      ", len(parsed["notes"]))
        print("Notes written:   ", len(transformed_notes))
        print("Block stats:     ", stats)

        if parsed["unmatched_note_on"] or parsed["unmatched_note_off"]:
            print(
                "Warning: unmatched MIDI note messages:",
                "note_on =", parsed["unmatched_note_on"],
                ", note_off =", parsed["unmatched_note_off"],
            )

        print(
            "Note: non-note MIDI events (tempo, pedal, controllers, etc.) "
            "remain at their original times."
        )

    return {
        "output": output_filename,
        "action": action,
        "stats": stats,
        "notes_in": len(parsed["notes"]),
        "notes_out": len(transformed_notes),
    }


# -----------------------------------------------------------------------------
# Example: Für Elise
# -----------------------------------------------------------------------------
# Uncomment and adapt the path:
#
G = DihedralGroup(4)
#
result = apply_group_to_midi(
    "fuer_elise.mid",
    "fuer_elise_D4_permuted.mid",
     G,
     representation="regular",   # |D4| = 8 -> 8 onset slots per block
     side="left",
    scope="global",
     identity_first=True,
     partial_block="keep",
     inverse=False,
     verbose=True,
 )
#
# For a permutation group with its NATURAL action instead:
#
# G = SymmetricGroup(4)
# apply_group_to_midi(
#     "fuer_elise.mid",
#     "fuer_elise_S4_natural.mid",
#     G,
#     representation="given",     # degree 4 -> 4 onset slots per block
# )
