#!/usr/bin/env python3
"""Literature-informed modal piano synthesizer: MIDI -> WAV/MP3.

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

The inharmonic, damped modes and the spectral role of the strike are based on
published physical acoustics, while the numerical constants below are artistic
un-calibrated defaults. This is a reduced model, NOT a detailed hammer contact
solver, full bridge coupling, FEM soundboard, or a clone of Pianoteq.
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


def inharmonic_frequency(f1, B, n):
    """Use measured first partial f1, not the ideal flexible-string f0."""
    return n*f1*math.sqrt((1 + B*n*n)/(1+B))


def inharmonicity(pitch):
    """Illustrative U-shaped per-key stiffness curve; NOT measured piano B."""
    x = (pitch-21)/87.
    return 0.00009 + 0.0025*max(0., 0.45-x)**2 + 0.0020*max(0., x-.5)**2


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


def synthesize(notes, length, sr, partials, release, board_mix, room, gain, verbose):
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
        contact = 0.00175-0.00092*v
        attack = 0.0015 + .001*(1-v)
        strike = .135  # effective hammer position along speaking string
        # Slower lower modes and more rapidly damped upper modes.
        register = np.clip((note.pitch-21)/87., 0., 1.)
        tau0 = 2.1 + 8.4*(1-register)**1.6
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
    ap = argparse.ArgumentParser(description='Literature-informed sample-free modal piano MIDI renderer.')
    ap.add_argument('midi',type=Path)
    ap.add_argument('output',type=Path,help='WAV or MP3')
    ap.add_argument('--sample-rate',type=int,default=32000)
    ap.add_argument('--partials',type=int,default=16)
    ap.add_argument('--release',type=float,default=.16,help='Damper e-folding time in seconds')
    ap.add_argument('--soundboard',type=float,default=.30,help='0 disables simplified modal soundboard')
    ap.add_argument('--room',type=float,default=.025,help='Optional subtle early reflections; 0 dry')
    ap.add_argument('--gain',type=float,default=1.)
    ap.add_argument('--bitrate',type=int,default=192,help='MP3 kbps')
    ap.add_argument('--seconds',type=float,default=None,help='Render preview of first N seconds')
    ap.add_argument('--tail',type=float,default=2.0)
    ap.add_argument('--verbose',action='store_true')
    args=ap.parse_args()
    if not (8000<=args.sample_rate<=192000 and 1<=args.partials<=64 and
            0.01<=args.release<=4 and 0<=args.soundboard<=2 and 0<=args.room<=1 and
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
    print(f'MIDI: {args.midi}; notes: {len(notes)}, input duration: {end:.3f}s, output: {total:.3f}s', flush=True)
    audio=synthesize(notes,total,args.sample_rate,args.partials,args.release,
                     args.soundboard,args.room,args.gain,args.verbose)
    write_audio(args.output,audio,args.sample_rate,args.bitrate)
    print(f'written: {args.output} ({len(audio)/args.sample_rate:.3f}s stereo)',flush=True)

if __name__=='__main__':
    main()
