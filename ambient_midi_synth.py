#!/usr/bin/env python3
"""Ambient MIDI synthesizer: MIDI -> generative stereo pads -> WAV/MP3.

No SoundFont, MIDIUtil, FluidSynth, or external synthesizer is required.
Only mido, numpy, and numba are needed; FFmpeg is needed for MP3 output.

The renderer preserves MIDI note pitch, onset, length, velocity, channel, and
all tempo changes. Sustain pedal (CC64) is interpreted. It deliberately maps
all pitched instruments to its own ambient timbre, ignoring program changes.

Examples:
    python ambient_midi_synth.py input.mid ambient.mp3
    python ambient_midi_synth.py input.mid preview.wav --seconds 45 --preset deep
    python ambient_midi_synth.py input.mid ambient.mp3 --preset glass --wet 0.56
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import mido
import numpy as np
from numba import njit


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: int
    channel: int


PRESETS = {
    'cloud': {'attack': 0.075, 'release': 1.85, 'wet': 0.48,
              'detune': 5.5, 'brightness': 0.30, 'feedback': 0.85},
    'deep':  {'attack': 0.14, 'release': 3.00, 'wet': 0.62,
              'detune': 4.0, 'brightness': 0.16, 'feedback': 0.89},
    'glass': {'attack': 0.038, 'release': 2.35, 'wet': 0.55,
              'detune': 7.0, 'brightness': 0.55, 'feedback': 0.87},
}


def read_midi_notes(path: Path, *, skip_drums: bool = True):
    """Read merged tracks and convert event delta ticks using the active tempo.

    Note-offs match note-ons FIFO for repeated (channel, pitch) pairs. Sustain
    pedal events postpone note endings until CC64 is released.
    """
    mf = mido.MidiFile(str(path))
    if mf.ticks_per_beat <= 0:
        raise ValueError('SMPTE time division is not supported')

    now, tempo = 0.0, 500_000  # microseconds per quarter note
    notes: list[Note] = []
    active = defaultdict(deque)
    sustained = defaultdict(list)
    pedal = defaultdict(bool)
    skipped = 0

    def finish(index, sec):
        note = notes[index]
        note.end = max(note.start + 0.01, sec)

    for msg in mido.merge_tracks(mf.tracks):
        now += mido.tick2second(msg.time, mf.ticks_per_beat, tempo)
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            continue
        if msg.is_meta:
            continue
        ch = getattr(msg, 'channel', 0)
        if skip_drums and ch == 9:
            if msg.type == 'note_on' and msg.velocity > 0:
                skipped += 1
            continue
        if msg.type == 'note_on' and msg.velocity > 0:
            notes.append(Note(now, now, msg.note, msg.velocity, ch))
            active[(ch, msg.note)].append(len(notes) - 1)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            queue = active[(ch, msg.note)]
            if queue:
                idx = queue.popleft()
                if pedal[ch]:
                    sustained[ch].append(idx)
                else:
                    finish(idx, now)
        elif msg.type == 'control_change':
            if msg.control == 64:
                new_state = msg.value >= 64
                if pedal[ch] and not new_state:
                    for idx in sustained[ch]:
                        finish(idx, now)
                    sustained[ch].clear()
                pedal[ch] = new_state
            elif msg.control in (120, 123):
                for (key_ch, pitch), q in active.items():
                    if key_ch == ch:
                        while q:
                            finish(q.popleft(), now)
                for idx in sustained[ch]:
                    finish(idx, now)
                sustained[ch].clear()

    for q in active.values():
        while q:
            finish(q.popleft(), now)
    for held in sustained.values():
        for idx in held:
            finish(idx, now)
    notes.sort(key=lambda x: (x.start, x.channel, x.pitch))
    return notes, now, skipped


def make_wavetables(sr: int, table_len: int, brightness: float) -> np.ndarray:
    """Band-limit each note's soft additive wavetable below Nyquist."""
    t = 2.0 * np.pi * np.arange(table_len, dtype=np.float64) / table_len
    tables = np.zeros((128, table_len), dtype=np.float32)
    for p in range(128):
        freq = 440 * 2.0 ** ((p - 69) / 12.0)
        weights = [1.0, 0.22 + 0.22*brightness,
                   0.06 + 0.16*brightness,
                   0.025 + 0.11*brightness,
                   0.008 + 0.05*brightness]
        x = np.zeros(table_len, dtype=np.float64)
        for harmonic, weight in enumerate(weights, 1):
            if harmonic * freq < 0.44 * sr:
                x += weight * np.sin(harmonic*t + 0.13*harmonic)
        # All pitches normalize independently; loudness still follows velocity.
        x /= max(1.0, np.max(np.abs(x)))
        tables[p] = x.astype(np.float32)
    return tables


@njit(cache=True)
def _lookup(tab, phase, size):
    pos = phase
    i = int(pos)
    frac = pos - i
    j = i + 1
    if j == size:
        j = 0
    return tab[i] + (tab[j] - tab[i]) * frac


@njit(cache=True)
def _render_notes(data, tables, left, right, sr, attack, release,
                  cents, gain, seed):
    """JIT-compiled event renderer: two detuned wavetable oscillators/note."""
    length = left.size
    table_len = tables.shape[1]
    attack_s = max(1, int(attack * sr))
    release_s = max(1, int(release * sr))
    detune_left = 2.0 ** (-cents / 1200.0)
    detune_right = 2.0 ** (cents / 1200.0)
    for ni in range(data.shape[0]):
        start = int(round(data[ni, 0] * sr))
        hold = max(1, int(round((data[ni, 1] - data[ni, 0]) * sr)))
        pitch = int(data[ni, 2])
        vel = data[ni, 3]
        channel = int(data[ni, 4])
        if start >= length or pitch < 0 or pitch > 127:
            continue
        n = min(hold + release_s, length - start)
        freq = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
        inc_l = freq * detune_left * table_len / sr
        inc_r = freq * detune_right * table_len / sr
        # Deterministic phase shifts: same pitches don't always start in phase.
        phase_l = float((ni*911 + seed*53) % table_len)
        phase_r = float((ni*613 + seed*97 + 421) % table_len)
        v = (vel / 127.0) ** 1.35
        amplitude = gain * v
        # Stable channel-dependent stereo positioning with slight note variation.
        pan = ((channel * 7 + pitch * 3) % 13 - 6) / 22.0
        g_l = np.sqrt(0.5 * (1.0 - pan))
        g_r = np.sqrt(0.5 * (1.0 + pan))
        tab = tables[pitch]
        # Use attack level at note-off to avoid envelope discontinuities.
        a_off = min(1.0, hold / attack_s)
        a_off = a_off * a_off * (3.0 - 2.0*a_off)
        for k in range(n):
            if k < hold:
                a = min(1.0, k / attack_s)
                env = a*a*(3.0 - 2.0*a)
            else:
                r = (k - hold) / release_s
                tail = 1.0 - r
                env = a_off * tail * tail
            # Gentle non-static movement without changing MIDI timing.
            breath = 0.93 + 0.07*np.sin(2.0*np.pi*(0.11+channel*0.013)*k/sr)
            wl = _lookup(tab, phase_l, table_len)
            wr = _lookup(tab, phase_r, table_len)
            amp = amplitude * env * breath
            left[start+k] += amp * (0.84*wl + 0.26*wr) * g_l
            right[start+k] += amp * (0.84*wr + 0.26*wl) * g_r
            phase_l += inc_l
            phase_r += inc_r
            if phase_l >= table_len:
                phase_l -= table_len
            if phase_r >= table_len:
                phase_r -= table_len


@njit(cache=True)
def _reverb_inplace(l, r, sr, wet, feedback):
    """Stereo multi-comb + diffusion allpass, with crossfeed tape-like echoes.

    All operations use finite-sized ring buffers: no external impulse response,
    no enormous Fourier-transform workspace, and no SoundFont.
    """
    comb_ms_l = np.array([30.1, 33.7, 37.1, 40.9, 44.5, 48.7])
    comb_ms_r = np.array([31.3, 34.9, 38.9, 42.1, 46.3, 50.3])
    delays_l = np.maximum(1, np.round(comb_ms_l*sr/1000).astype(np.int64))
    delays_r = np.maximum(1, np.round(comb_ms_r*sr/1000).astype(np.int64))
    buflen = int(max(np.max(delays_l), np.max(delays_r)))
    comb_l = np.zeros((6, buflen), dtype=np.float32)
    comb_r = np.zeros((6, buflen), dtype=np.float32)
    ptr_l = np.zeros(6, dtype=np.int64)
    ptr_r = np.zeros(6, dtype=np.int64)
    lp_l = np.zeros(6, dtype=np.float64)
    lp_r = np.zeros(6, dtype=np.float64)

    # Two stages of short diffusion allpasses, differently tuned per channel.
    ap_sizes_l = np.maximum(1, np.round(np.array([5.1, 1.7])*sr/1000).astype(np.int64))
    ap_sizes_r = np.maximum(1, np.round(np.array([5.7, 2.1])*sr/1000).astype(np.int64))
    ap_max = int(max(np.max(ap_sizes_l), np.max(ap_sizes_r)))
    ap_l = np.zeros((2, ap_max), dtype=np.float32)
    ap_r = np.zeros((2, ap_max), dtype=np.float32)
    ap_ptr_l = np.zeros(2, dtype=np.int64)
    ap_ptr_r = np.zeros(2, dtype=np.int64)

    d1 = max(1, int(.37*sr))
    d2 = max(1, int(.57*sr))
    echo_l = np.zeros(d1, dtype=np.float32)
    echo_r = np.zeros(d2, dtype=np.float32)
    ep1, ep2 = 0, 0
    dry_amt = 1.0 - 0.30*wet
    wet_amt = 0.55*wet

    for i in range(l.size):
        dl, dr = float(l[i]), float(r[i])
        oldl, oldr = echo_l[ep1], echo_r[ep2]
        echo_l[ep1] = dl*0.38 + oldr*0.24
        echo_r[ep2] = dr*0.38 + oldl*0.24
        ep1 += 1
        ep2 += 1
        if ep1 == d1: ep1 = 0
        if ep2 == d2: ep2 = 0
        incoming_l = 0.19 * (dl + 0.34*oldr)
        incoming_r = 0.19 * (dr + 0.34*oldl)
        cl, cr = 0.0, 0.0
        for j in range(6):
            k_l, k_r = ptr_l[j], ptr_r[j]
            old_l = float(comb_l[j, k_l])
            old_r = float(comb_r[j, k_r])
            lp_l[j] = 0.27*old_l + 0.73*lp_l[j]
            lp_r[j] = 0.27*old_r + 0.73*lp_r[j]
            comb_l[j, k_l] = incoming_l + feedback*lp_l[j]
            comb_r[j, k_r] = incoming_r + feedback*lp_r[j]
            cl += old_l
            cr += old_r
            ptr_l[j] = (k_l + 1) % delays_l[j]
            ptr_r[j] = (k_r + 1) % delays_r[j]
        cl /= 6.0
        cr /= 6.0
        for j in range(2):
            p_l, p_r = ap_ptr_l[j], ap_ptr_r[j]
            v_l, v_r = float(ap_l[j,p_l]), float(ap_r[j,p_r])
            ap_l[j,p_l] = cl + 0.50*v_l
            ap_r[j,p_r] = cr + 0.50*v_r
            cl = -cl*0.50 + v_l
            cr = -cr*0.50 + v_r
            ap_ptr_l[j] = (p_l + 1) % ap_sizes_l[j]
            ap_ptr_r[j] = (p_r + 1) % ap_sizes_r[j]
        l[i] = dry_amt*dl + wet_amt*cl + 0.09*wet*oldr
        r[i] = dry_amt*dr + wet_amt*cr + 0.09*wet*oldl


def encode_audio(audio: np.ndarray, out: Path, sr: int, mp3_bitrate='192k'):
    """Write PCM WAV using stdlib, or pipe PCM directly to FFmpeg for MP3."""
    out.parent.mkdir(parents=True, exist_ok=True)
    chunk = sr * 4
    if out.suffix.lower() == '.wav':
        with wave.open(str(out), 'wb') as fp:
            fp.setnchannels(2)
            fp.setsampwidth(2)
            fp.setframerate(sr)
            for i in range(0, len(audio), chunk):
                block = np.clip(audio[i:i+chunk], -1.0, 1.0)
                fp.writeframes((block * 32767).astype('<i2').tobytes())
    elif out.suffix.lower() == '.mp3':
        if not shutil.which('ffmpeg'):
            raise RuntimeError('FFmpeg is needed for MP3. Use a .wav output or install ffmpeg.')
        proc = subprocess.Popen([
            'ffmpeg','-y','-loglevel','error',
            '-f','f32le','-ar',str(sr),'-ac','2','-i','pipe:0',
            '-c:a','libmp3lame','-b:a',mp3_bitrate,
            '-id3v2_version','3',str(out)], stdin=subprocess.PIPE)
        try:
            for i in range(0, len(audio), chunk):
                proc.stdin.write(np.asarray(audio[i:i+chunk],dtype='<f4').tobytes())
            proc.stdin.close()
            rc = proc.wait()
            if rc:
                raise RuntimeError(f'FFmpeg failed, code {rc}')
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    else:
        raise ValueError('Output must end in .wav or .mp3')


def render(midi_path: Path, output_path: Path, *, preset='cloud', sample_rate=24000,
           attack=None, release=None, wet=None, detune=None, seed=42,
           seconds=None, tail=4.0, gain=0.039, mp3_bitrate='192k'):
    cfg = PRESETS[preset].copy()
    if attack is not None: cfg['attack'] = attack
    if release is not None: cfg['release'] = release
    if wet is not None: cfg['wet'] = wet
    if detune is not None: cfg['detune'] = detune
    notes, midi_end, skipped = read_midi_notes(midi_path)
    if not notes:
        raise ValueError('No pitched note events found in this MIDI.')
    if seconds is not None:
        notes = [n for n in notes if n.start < seconds]
        for n in notes:
            n.end = min(n.end, seconds)
        if not notes:
            raise ValueError('No notes before --seconds.')
        musical_end = min(seconds, midi_end)
    else:
        musical_end = max(midi_end, max(n.end for n in notes))
    length = int(np.ceil((musical_end + max(tail, cfg['release']+0.5)) * sample_rate))
    left = np.zeros(length, dtype=np.float32)
    right = np.zeros(length, dtype=np.float32)
    data = np.array([[n.start, n.end, n.pitch, n.velocity, n.channel] for n in notes],
                    dtype=np.float64)
    print(f'MIDI: {midi_path.name}; notes={len(notes)}; duration={musical_end:.2f}s; '
          f'skipped percussion notes={skipped}', flush=True)
    print(f'Preset: {preset} | attack={cfg["attack"]:.3f}s '
          f'release={cfg["release"]:.3f}s wet={cfg["wet"]:.2f} '
          f'detune=+/-{cfg["detune"]:.1f} cents | {sample_rate} Hz', flush=True)
    tables = make_wavetables(sample_rate, 2048, cfg['brightness'])
    print('Synthesizing ambient pad voices...', flush=True)
    _render_notes(data, tables, left, right, sample_rate, cfg['attack'],
                  cfg['release'], cfg['detune'], gain, seed)
    print('Adding stereo echoes and algorithmic reverb...', flush=True)
    _reverb_inplace(left, right, sample_rate, cfg['wet'], cfg['feedback'])
    audio = np.empty((length,2), dtype=np.float32)
    audio[:,0], audio[:,1] = left, right
    del left, right
    # Smooth overloads, preserve stereo width, and set safe peak level.
    np.tanh(audio * 1.20, out=audio)
    peak = float(np.max(np.abs(audio)))
    if peak:
        audio *= 0.945/peak  # about -0.49 dBTP sample peak
    print(f'Writing {output_path} ({length/sample_rate:.1f}s)...', flush=True)
    encode_audio(audio, output_path, sample_rate, mp3_bitrate)
    return {'notes':len(notes), 'duration':length/sample_rate, 'peak':peak,
            'sample_rate':sample_rate, 'bytes':output_path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('midi', type=Path, help='Input MIDI file')
    parser.add_argument('output', type=Path, help='Output .wav or .mp3')
    parser.add_argument('--preset', choices=PRESETS, default='cloud')
    parser.add_argument('--sample-rate', type=int, default=24000)
    parser.add_argument('--attack', type=float, default=None, help='Pad fade-in in seconds')
    parser.add_argument('--release', type=float, default=None, help='Note fade-out in seconds')
    parser.add_argument('--wet', type=float, default=None, help='Reverb amount 0..1')
    parser.add_argument('--detune', type=float, default=None, help='Oscillator separation in cents')
    parser.add_argument('--seconds', type=float, default=None, help='Render only first N musical seconds')
    parser.add_argument('--tail', type=float, default=4.0, help='Time after music to let echoes fade')
    parser.add_argument('--gain', type=float, default=0.039, help='Voice gain before soft limiting')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mp3-bitrate', default='192k')
    args = parser.parse_args()
    if args.sample_rate < 16000 or args.sample_rate > 96000:
        parser.error('--sample-rate must be between 16000 and 96000')
    if args.seconds is not None and args.seconds <= 0:
        parser.error('--seconds must be positive')
    if args.wet is not None and not 0 <= args.wet <= 1:
        parser.error('--wet must be between 0 and 1')
    for name in ['attack','release','detune']:
        value = getattr(args,name)
        if value is not None and value <= 0:
            parser.error(f'--{name} must be positive')
    if not args.midi.is_file():
        parser.error(f'Input not found: {args.midi}')
    stats = render(args.midi, args.output, preset=args.preset,
                   sample_rate=args.sample_rate, attack=args.attack,
                   release=args.release, wet=args.wet, detune=args.detune,
                   seconds=args.seconds, tail=args.tail, gain=args.gain,
                   seed=args.seed, mp3_bitrate=args.mp3_bitrate)
    print(f'Done: {stats}', flush=True)


if __name__ == '__main__':
    main()
