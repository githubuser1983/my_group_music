#!/usr/bin/env python3
"""Measurement-anchored modal piano synthesizer: MIDI -> WAV/MP3.

No samples, SoundFont, Pianoteq, MIDIUtil or pretrained weights are used.
Dependencies: mido, numpy, numba, scipy; ffmpeg for MP3.

Scientific basis (independent implementation, not a reproduction of their code):
  Bank, Zambon, Fontana (2010), A Modal-Based Real-Time Piano Synthesizer,
    IEEE TASLP 18(4), 809-821, doi:10.1109/TASL.2010.2040524.
  Simionato, Fasciani, Holm (2024), Physics-informed differentiable method
    for piano modeling, Frontiers in Signal Processing,
    doi:10.3389/frsip.2023.1276748.
  Elie, Cotte, Boutillon (2022), Physically-based sound synthesis software
    for Computer-Aided-Design of piano soundboards,
    Acta Acustica 6, 30, doi:10.1051/aacus/2022024.
  de Paula, Smith, Valimaki, Reiss (2026), Four Decades of Digital Waveguides,
    JAES 74, 464-484, doi:10.17743/jaes.2026.0283 (methodological context).

The five inharmonicity measurements and five overall unpedaled T60 values
are paired example tones in Lehtonen, Penttinen, Rauhala & Valimaki (2007),
JASA 122:1787-1797, Tables I-II, DOI 10.1121/1.2756172. For non-anchor
keys and velocities log interpolation/clamping is used (estimates). Hammer
contact endpoints come from Askenfelt & Jansson (1990), on another piano.
The measured overall T60 is used as a PROXY for the first modal decay, not
a claim that the first partial itself has this measured T60.
Remaining modal spectral/decay constants, strike position, unison detuning,
post-note-off damper release and panning are model choices, not measurements.
The result is thus measurement-ANCHORED, not a digital twin of any piano.
No permanent room reflection or soundboard resonator by default. An optional
subtle room bus is fed only while MIDI CC64 sustain is engaged, with a smooth
send transition and a short tail continuing after pedal release. These room
values are modeled aesthetic settings, not empirical measurements. The output still uses
the original piano synth architecture, not Pianoteq code/samples.
"""
from __future__ import annotations
import argparse
import math
import shutil
import subprocess
import sys
import wave
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np
from numba import njit
from scipy.signal import iirpeak, sosfilt, tf2sos


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: int
    channel: int


def read_midi(path: Path, skip_drums=True):
    """Use current tempo throughout merged track; handle sustain CC64 per channel."""
    mid = mido.MidiFile(str(path))
    if mid.ticks_per_beat <= 0:
        raise ValueError('SMPTE MIDI division is not supported')
    time_s, tempo = 0., 500000
    active = defaultdict(deque)
    pedal = defaultdict(bool)
    held = defaultdict(list)
    notes = []

    def release(idx, t):
        notes[idx].end = max(notes[idx].start + .002, t)

    for msg in mido.merge_tracks(mid.tracks):
        time_s += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            continue
        if msg.is_meta:
            continue
        c = getattr(msg, 'channel', 0)
        if skip_drums and c == 9:
            continue
        if msg.type == 'note_on' and msg.velocity > 0:
            active[(c, msg.note)].append(len(notes))
            notes.append(Note(time_s, time_s, msg.note, msg.velocity, c))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if active[(c, msg.note)]:
                idx = active[(c, msg.note)].popleft()
                if pedal[c]:
                    held[c].append(idx)
                else:
                    release(idx, time_s)
        elif msg.type == 'control_change':
            if msg.control == 64:
                new = msg.value >= 64
                if pedal[c] and not new:
                    for idx in held[c]:
                        release(idx, time_s)
                    held[c].clear()
                pedal[c] = new
            elif msg.control in (120, 123):
                for (ch, _), q in active.items():
                    if ch == c:
                        while q:
                            release(q.popleft(), time_s)
                for idx in held[c]:
                    release(idx, time_s)
                held[c].clear()
    for q in active.values():
        while q:
            release(q.popleft(), time_s)
    for h in held.values():
        for idx in h:
            release(idx, time_s)
    notes.sort(key=lambda n: (n.start, n.channel, n.pitch))
    return notes, time_s


def read_pedal_intervals(path: Path):
    """Read CC64 pedal-down windows in seconds, using all MIDI tempo events.

    Intervals from distinct channels are combined for a modest global stereo
    pedal-room send. Individual note holding is separately honored by
    read_midi().
    """
    mid = mido.MidiFile(str(path))
    time_s, tempo = 0., 500000
    down_since = {}
    intervals = []
    down_count = 0
    for msg in mido.merge_tracks(mid.tracks):
        time_s += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        if msg.type != 'control_change' or msg.control != 64:
            continue
        c = msg.channel
        if msg.value >= 64 and c not in down_since:
            down_since[c] = time_s
            down_count += 1
        elif msg.value < 64 and c in down_since:
            intervals.append((down_since.pop(c), time_s))
    for t0 in down_since.values():
        intervals.append((t0, time_s))
    intervals.sort()
    # Union overlapping intervals across MIDI channels for the shared room bus.
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged, down_count


@njit(cache=True)
def _smooth_pedal_gate(gate, attack_coeff, release_coeff):
    out = np.zeros(gate.size, np.float32)
    value = 0.
    for k in range(gate.size):
        target = gate[k]
        coeff = attack_coeff if target > value else release_coeff
        value = target + coeff * (value - target)
        out[k] = value
    return out


def pedal_send_envelope(intervals, samples, sample_rate, attack=.025, release=.11):
    gate = np.zeros(samples, np.float32)
    for start, end in intervals:
        lo = max(0, min(samples, int(round(start * sample_rate))))
        hi = max(lo, min(samples, int(round(end * sample_rate))))
        gate[lo:hi] = 1.
    ac = math.exp(-1. / max(1., attack * sample_rate))
    rc = math.exp(-1. / max(1., release * sample_rate))
    return _smooth_pedal_gate(gate, ac, rc)


@njit(cache=True)
def _pedal_room_bus(left, right, envelope, sample_rate, rt60):
    """Short, dark, stereo comb/diffuser bus, excited only by CC64-down audio.

    Its coefficients are a deliberate aesthetic setting, not measured
    resonances from the cited instrument. State continues to ring after
    pedal-up; send stops smoothly when the damper returns.
    """
    n = left.size
    # Mutually different short delay times avoid one perceptible discrete echo.
    delays_l = np.array([.0297, .0371, .0411, .0531])
    delays_r = np.array([.0313, .0397, .0469, .0587])
    dl = np.empty(4, np.int64)
    dr = np.empty(4, np.int64)
    for j in range(4):
        dl[j] = max(1, int(round(delays_l[j] * sample_rate)))
        dr[j] = max(1, int(round(delays_r[j] * sample_rate)))
    maxdelay = max(np.max(dl), np.max(dr))
    buffer_l = np.zeros((4, maxdelay + 1), np.float32)
    buffer_r = np.zeros((4, maxdelay + 1), np.float32)
    at_l = np.zeros(4, np.int64)
    at_r = np.zeros(4, np.int64)
    damping_l = np.zeros(4, np.float64)
    damping_r = np.zeros(4, np.float64)
    feedback_l = np.empty(4, np.float64)
    feedback_r = np.empty(4, np.float64)
    for j in range(4):
        # 60 dB amplitude reduction at rt60 for a free-running comb.
        feedback_l[j] = 10.**(-3. * dl[j] / (sample_rate * rt60))
        feedback_r[j] = 10.**(-3. * dr[j] / (sample_rate * rt60))
    wet_l = np.zeros(n, np.float32)
    wet_r = np.zeros(n, np.float32)
    for k in range(n):
        send_l = left[k] * envelope[k]
        send_r = right[k] * envelope[k]
        accum_l = 0.
        accum_r = 0.
        for j in range(4):
            val_l = buffer_l[j, at_l[j]]
            val_r = buffer_r[j, at_r[j]]
            # Damped feedback removes high-frequency chatter from the tail.
            damping_l[j] = .56 * damping_l[j] + .44 * val_l
            damping_r[j] = .56 * damping_r[j] + .44 * val_r
            buffer_l[j, at_l[j]] = send_l + feedback_l[j] * damping_l[j]
            buffer_r[j, at_r[j]] = send_r + feedback_r[j] * damping_r[j]
            accum_l += damping_l[j]
            accum_r += damping_r[j]
            at_l[j] += 1
            at_r[j] += 1
            if at_l[j] >= dl[j]:
                at_l[j] = 0
            if at_r[j] >= dr[j]:
                at_r[j] = 0
        wet_l[k] = accum_l * .25
        wet_r[k] = accum_r * .25
    return wet_l, wet_r


def inharmonic_frequency(f1, B, n):
    """Use measured first partial f1, not the ideal flexible-string f0."""
    return n*f1*math.sqrt((1 + B*n*n)/(1+B))


# Lehtonen et al., 2007, Table I (same set of five test tones as Table II).
# C2, C3, C4, D5, C6: B from recorded piano tones.
# These ARE reported fitted inharmonicity coefficients; NOT arbitrary defaults.
MEASURED_B = {36: 3.8e-5, 48: 1.1e-4, 60: 3.3e-4,
              74: 1.2e-3, 84: 2.3e-3}
# Table II of that SAME paper: overall -60 dB decay with pedal NOT pressed,
# NOT damper decay after the note is released.
MEASURED_T60_S = {36: 20.3, 48: 12.5, 60: 9.5, 74: 14.7, 84: 12.0}


def _log_interpolated(pitch, anchors):
    """Log-linearly interpolate positive measured anchors; clamp outside.

    The value for an unmeasured key is an ESTIMATE, not a measurement.
    """
    keys = sorted(anchors)
    if pitch <= keys[0]: return float(anchors[keys[0]])
    if pitch >= keys[-1]: return float(anchors[keys[-1]])
    for lo,hi in zip(keys,keys[1:]):
        if lo <= pitch <= hi:
            t=(pitch-lo)/(hi-lo)
            return math.exp((1-t)*math.log(anchors[lo])+t*math.log(anchors[hi]))
    raise AssertionError('bad key interpolation')


def inharmonicity(pitch):
    """Measured five-note B curve (Lehtonen et al. Table I), log interpolation."""
    return _log_interpolated(pitch, MEASURED_B)


def fundamental_decay_time(pitch):
    """Convert measured total amplitude T60 to a modal e-folding proxy.

    Total tone T60 is NOT the same as an individual partial's T60. Using it
    for the fundamental is an approximate synthesis mapping, clearly not a
    direct modal measurement. Higher partials retain parametric loss law.
    """
    return _log_interpolated(pitch, MEASURED_T60_S)/math.log(1000.)


def hammer_contact_time(pitch, v):
    """Interpolate measured contact-time trends (Askenfelt/Jansson).

    Empirical endpoints approx 4 ms bass to <1 ms highest treble; around
    +/-20 percent with playing level. Exact per-key mapping is modeled.
    """
    x=max(0.,min(1.,(pitch-21)/87.))
    baseline=.0040*(.0008/.0040)**x
    return baseline*(1.2-.4*max(0.,min(1.,v)))


@njit(cache=True)
def _add_mode(left, right, first, ns, duration, frequency, decay, damp,
              amp_l, amp_r, sample_rate, attack):
    # Damped modal sinusoid. Per-mode recursive quadrature avoids trig/sample.
    omega = 2*math.pi*frequency/sample_rate
    si, co = 0., 1.
    st, ct = math.sin(omega), math.cos(omega)
    decay_step = math.exp(-1.0/(sample_rate*decay))
    damp_step = math.exp(-1.0/(sample_rate*damp))
    a = 1.
    rel = 1.
    k_off = int(max(1, round(duration*sample_rate)))
    at_n = max(1, int(attack*sample_rate))
    n = min(ns, left.size-first)
    for k in range(n):
        if k > 0:
            tmp = si*ct + co*st
            co = co*ct - si*st
            si = tmp
            # Prevent numerical drift in the rotation over multi-second tails.
            if (k & 4095) == 0:
                norm = math.sqrt(si*si+co*co)
                si /= norm
                co /= norm
        if k >= k_off:
            rel *= damp_step
        # Smooth beginning, spectral envelope is already attack/velocity dependent.
        onset = (k/at_n) if k < at_n else 1.
        y = si*a*rel*onset
        left[first+k] += amp_l*y
        right[first+k] += amp_r*y
        a *= decay_step


def synthesize(notes, length, sr, partials, release, board_mix, room, gain, verbose,
               pedal_envelope=None, pedal_wet=0., pedal_rt60=.65):
    L = np.zeros(int(math.ceil(length*sr)), np.float32)
    R = np.zeros_like(L)
    all_notes = len(notes)
    for j, note in enumerate(notes):
        first = int(round(note.start*sr))
        if first >= L.size:
            continue
        f1 = 440.*2**((note.pitch-69)/12.)
        B = inharmonicity(note.pitch)
        v = max(.01, note.velocity/127.)
        # Impact-force bandwidth increases with hammer velocity: reduced proxy,
        # not a numerically solved nonlinear felt-string collision.
        contact = hammer_contact_time(note.pitch, v)
        attack = 0.0015 + .001*(1-v)
        strike = .135  # effective hammer position along speaking string
        # Slower lower modes and more rapidly damped upper modes.
        register = np.clip((note.pitch-21)/87., 0., 1.)
        tau0 = fundamental_decay_time(note.pitch)
        damper_tau = release*(1.20-0.35*register)
        duration = note.end-note.start
        # Limit tail according to damper time and natural decay.
        tail = min(3.5, max(.45, damper_tau*7))
        ns = int(math.ceil((duration+tail)*sr))
        # Bass is single-string, tenor double, treble three-string unison.
        strings = 1 if note.pitch < 40 else (2 if note.pitch < 57 else 3)
        cents = (0.,) if strings == 1 else ((-.62,.62) if strings == 2 else (-.82,0.,.93))
        pan = np.clip((note.pitch-64)/85., -.43,.43)
        # Actual overall amplitude is balanced at final peak normalization.
        velocity_amp = (v**1.5)*gain/np.sqrt(strings)
        for s, detune in enumerate(cents):
            fstring = f1*2**(detune/1200.)
            pan_s = float(np.clip(pan+(s-(strings-1)/2)*.035, -.62,.62))
            gl = math.sqrt(.5*(1-pan_s))
            gr = math.sqrt(.5*(1+pan_s))
            for k in range(1, partials+1):
                fk = inharmonic_frequency(fstring, B, k)
                if fk >= sr*.46:
                    break
                # Position-dependent modal excitation. A smooth contact-time
                # envelope attenuates the highest partials at low velocity.
                strike_amp = abs(math.sin(math.pi*k*strike))
                hammer = math.exp(-(.32*fk*contact)**2)
                spectral = strike_amp*hammer/k**.94
                if spectral < 5e-5:
                    continue
                decay = tau0/(1 + .065*k**1.42)*(1+(.12 if s == 0 else -.09 if s == 2 else 0.))
                ampl = velocity_amp*spectral*.065
                _add_mode(L,R,first,ns,duration,fk,decay,damper_tau,
                          ampl*gl,ampl*gr,sr,attack)
        if verbose and (j+1)%500 == 0:
            print(f'  rendered {j+1}/{all_notes} notes', flush=True)
    # Parallel damped second-order filters approximate selected soundboard
    # radiation modes; they are not a measured bridge admittance / FEM plate.
    if board_mix > 0:
        frequencies = [85,145,245,395,625,995,1580,2450,3650]
        qs =          [7,  9, 11, 13, 16, 18, 22,  28,  36]
        weights =     [.15,.20,.21,.18,.14,.11,.09,.07,.04]
        dry_l = L.copy()
        dry_r = R.copy()
        for f,q,w in zip(frequencies,qs,weights):
            if f >= sr*.45:
                continue
            b,a = iirpeak(f, q, fs=sr)
            sos = tf2sos(b,a)
            # Parallel soundboard modes driven by the ORIGINAL string mixture.
            L += board_mix*w*sosfilt(sos,dry_l).astype(np.float32)
            R += board_mix*w*sosfilt(sos,dry_r).astype(np.float32)
    if room > 0:
        # Tiny early reflection only; optional and intentionally restrained.
        for seconds, amount in ((.021,.27),(.037,.19),(.061,.11)):
            delay = int(seconds*sr)
            if delay < L.size:
                # copy source so stereo crossfeed isn't recursively overwritten
                ll = L[:-delay].copy()
                rr = R[:-delay].copy()
                L[delay:] += room*amount*rr
                R[delay:] += room*amount*ll
    if pedal_envelope is not None and pedal_wet > 0:
        wet_l, wet_r = _pedal_room_bus(L, R, pedal_envelope, sr, pedal_rt60)
        # Modest diffuse stereo crossfeed, no permanent room effect on dry notes.
        L += pedal_wet * (.84 * wet_l + .16 * wet_r)
        R += pedal_wet * (.84 * wet_r + .16 * wet_l)
    peak = max(float(np.max(np.abs(L))),float(np.max(np.abs(R))),1e-10)
    scale = .90/peak
    print(f'peak before normalization: {peak:.5f}; gain: {scale:.3f}', flush=True)
    return np.column_stack((L*scale,R*scale))


def write_audio(path, audio, sr, kbps):
    path.parent.mkdir(parents=True,exist_ok=True)
    # TPDF dither at 16-bit conversion, deterministic seed.
    rng = np.random.default_rng(1286)
    if path.suffix.lower() == '.wav':
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr)
            for k in range(0,len(audio),sr):
                block = audio[k:k+sr]
                noise = (rng.random(block.shape)-rng.random(block.shape))/65536.
                wf.writeframes(np.clip(np.rint((block+noise)*32767),-32768,32767).astype('<i2').tobytes())
    elif path.suffix.lower() == '.mp3':
        if not shutil.which('ffmpeg'):
            raise RuntimeError('ffmpeg required for MP3 output; use .wav otherwise')
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','f32le',
             '-ar',str(sr),'-ac','2','-i','pipe:0','-c:a','libmp3lame',
             '-b:a',f'{kbps}k',str(path)]
        proc = subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for k in range(0,len(audio),sr):
                proc.stdin.write(np.asarray(audio[k:k+sr],dtype='<f4').tobytes())
            proc.stdin.close()
            rc=proc.wait()
        except Exception:
            proc.kill(); proc.wait(); raise
        if rc:
            raise RuntimeError(f'ffmpeg exited with status {rc}')
    else:
        raise ValueError('Output must end in .wav or .mp3')


def main():
    ap = argparse.ArgumentParser(description='Measurement-anchored sample-free modal piano MIDI renderer.')
    ap.add_argument('midi',type=Path)
    ap.add_argument('output',type=Path,help='WAV or MP3')
    ap.add_argument('--sample-rate',type=int,default=32000)
    ap.add_argument('--partials',type=int,default=16)
    ap.add_argument('--release',type=float,default=.06,help='MODELED damper e-folding time in seconds; not a measured parameter')
    ap.add_argument('--soundboard',type=float,default=0.0,help='0 disables UNCALIBRATED simplified modal soundboard')
    ap.add_argument('--room',type=float,default=0.0,help='Optional artificial early reflections; 0 dry')
    ap.add_argument('--pedal-wet',type=float,default=.11,
                    help='Extra room send ONLY while sustain CC64 is down; 0 disables it')
    ap.add_argument('--pedal-rt60',type=float,default=.65,
                    help='Approximate added-room RT60 in seconds (modeled, not measured)')
    ap.add_argument('--pedal-attack',type=float,default=.025,
                    help='CC64 room-send fade-in in seconds')
    ap.add_argument('--pedal-release',type=float,default=.11,
                    help='CC64 room-send fade-out in seconds; reverb tail continues')
    ap.add_argument('--gain',type=float,default=1.)
    ap.add_argument('--bitrate',type=int,default=192,help='MP3 kbps')
    ap.add_argument('--seconds',type=float,default=None,help='Render preview of first N seconds')
    ap.add_argument('--tail',type=float,default=2.0)
    ap.add_argument('--verbose',action='store_true')
    args=ap.parse_args()
    if not (8000<=args.sample_rate<=192000 and 1<=args.partials<=64 and
            0.01<=args.release<=4 and 0<=args.soundboard<=2 and 0<=args.room<=1 and
            0<=args.pedal_wet<=1 and .1<=args.pedal_rt60<=3 and
            .001<=args.pedal_attack<=2 and .001<=args.pedal_release<=3 and
            0<args.gain<=10 and args.tail>=0 and args.bitrate>0):
        ap.error('Invalid synthesis parameters')
    notes,end=read_midi(args.midi)
    if args.seconds is not None:
        if args.seconds<=0: ap.error('--seconds must be positive')
        notes=[n for n in notes if n.start<args.seconds]
        for n in notes:
            n.end=min(n.end,args.seconds)
        end=min(end,args.seconds)
    if not notes:
        raise ValueError('No pitched MIDI notes found')
    total=end+args.tail
    pedal_intervals, pedal_presses = read_pedal_intervals(args.midi)
    pedal_envelope = (pedal_send_envelope(pedal_intervals, int(math.ceil(total*args.sample_rate)),
                     args.sample_rate, args.pedal_attack, args.pedal_release)
                     if args.pedal_wet > 0 and pedal_intervals else None)
    print(f'MIDI: {args.midi}; notes: {len(notes)}, input duration: {end:.3f}s, output: {total:.3f}s', flush=True)
    print(f'CC64 presses: {pedal_presses}; combined pedal-down intervals: {len(pedal_intervals)}; '
          f'pedal wet: {args.pedal_wet:.3f}; RT60: {args.pedal_rt60:.2f}s', flush=True)
    audio=synthesize(notes,total,args.sample_rate,args.partials,args.release,
                     args.soundboard,args.room,args.gain,args.verbose,
                     pedal_envelope,pedal_wet=args.pedal_wet,pedal_rt60=args.pedal_rt60)
    write_audio(args.output,audio,args.sample_rate,args.bitrate)
    print(f'written: {args.output} ({len(audio)/args.sample_rate:.3f}s stereo)',flush=True)

if __name__=='__main__':
    main()
