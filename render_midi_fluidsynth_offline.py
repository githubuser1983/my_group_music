#!/usr/bin/env python3
import argparse
import ctypes
import subprocess
from pathlib import Path
import mido

LIBNAME = 'libfluidsynth.so.3'


def _load_lib():
    lib = ctypes.CDLL(LIBNAME)
    vp = ctypes.c_void_p
    ci = ctypes.c_int
    cd = ctypes.c_double
    cp = ctypes.c_char_p

    lib.new_fluid_settings.restype = vp
    lib.delete_fluid_settings.argtypes = [vp]
    lib.fluid_settings_setnum.argtypes = [vp, cp, cd]
    lib.fluid_settings_setnum.restype = ci
    lib.fluid_settings_setint.argtypes = [vp, cp, ci]
    lib.fluid_settings_setint.restype = ci

    lib.new_fluid_synth.argtypes = [vp]
    lib.new_fluid_synth.restype = vp
    lib.delete_fluid_synth.argtypes = [vp]

    lib.fluid_synth_sfload.argtypes = [vp, cp, ci]
    lib.fluid_synth_sfload.restype = ci

    lib.fluid_synth_noteon.argtypes = [vp, ci, ci, ci]
    lib.fluid_synth_noteon.restype = ci
    lib.fluid_synth_noteoff.argtypes = [vp, ci, ci]
    lib.fluid_synth_noteoff.restype = ci
    lib.fluid_synth_cc.argtypes = [vp, ci, ci, ci]
    lib.fluid_synth_cc.restype = ci
    lib.fluid_synth_program_change.argtypes = [vp, ci, ci]
    lib.fluid_synth_program_change.restype = ci
    lib.fluid_synth_pitch_bend.argtypes = [vp, ci, ci]
    lib.fluid_synth_pitch_bend.restype = ci
    lib.fluid_synth_channel_pressure.argtypes = [vp, ci, ci]
    lib.fluid_synth_channel_pressure.restype = ci
    lib.fluid_synth_key_pressure.argtypes = [vp, ci, ci, ci]
    lib.fluid_synth_key_pressure.restype = ci

    lib.fluid_synth_write_s16.argtypes = [
        vp, ci,
        ctypes.POINTER(ctypes.c_short), ci, ci,
        ctypes.POINTER(ctypes.c_short), ci, ci,
    ]
    lib.fluid_synth_write_s16.restype = ci
    return lib


def render(midi_path, mp3_path, soundfont, sample_rate=44100, gain=0.55,
           mp3_quality=2, tail_seconds=4.0, chunk_frames=4096):
    midi_path = Path(midi_path)
    mp3_path = Path(mp3_path)
    soundfont = Path(soundfont)
    for p in (midi_path, soundfont):
        if not p.is_file():
            raise FileNotFoundError(p)

    mid = mido.MidiFile(str(midi_path))
    merged = mido.merge_tracks(mid.tracks)
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000  # 120 BPM default, microseconds per beat

    lib = _load_lib()
    settings = lib.new_fluid_settings()
    if not settings:
        raise RuntimeError('Could not create FluidSynth settings')
    synth = None
    ff = None

    try:
        lib.fluid_settings_setnum(settings, b'synth.sample-rate', float(sample_rate))
        lib.fluid_settings_setnum(settings, b'synth.gain', float(gain))
        lib.fluid_settings_setint(settings, b'synth.reverb.active', 0)
        lib.fluid_settings_setint(settings, b'synth.chorus.active', 0)
        lib.fluid_settings_setint(settings, b'synth.cpu-cores', 4)
        synth = lib.new_fluid_synth(settings)
        if not synth:
            raise RuntimeError('Could not create FluidSynth synth')
        sfid = lib.fluid_synth_sfload(synth, str(soundfont).encode(), 1)
        if sfid < 0:
            raise RuntimeError(f'Could not load SoundFont: {soundfont}')

        ff_cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 's16le', '-ar', str(sample_rate), '-ac', '2', '-i', 'pipe:0',
            '-c:a', 'libmp3lame', '-q:a', str(mp3_quality),
            '-id3v2_version', '3', str(mp3_path)
        ]
        ff = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)

        buf = (ctypes.c_short * (chunk_frames * 2))()
        ptr = ctypes.cast(buf, ctypes.POINTER(ctypes.c_short))
        frac_frames = 0.0
        rendered_frames = 0

        def write_audio(frames):
            nonlocal rendered_frames
            while frames > 0:
                n = min(frames, chunk_frames)
                rc = lib.fluid_synth_write_s16(synth, n, ptr, 0, 2, ptr, 1, 2)
                if rc != 0:
                    raise RuntimeError(f'FluidSynth write failed: {rc}')
                ff.stdin.write(ctypes.string_at(ptr, n * 2 * ctypes.sizeof(ctypes.c_short)))
                rendered_frames += n
                frames -= n

        for msg in merged:
            # Advance audio by this event's delta time using the tempo active
            # during the preceding interval.
            if msg.time:
                sec = mido.tick2second(msg.time, ticks_per_beat, tempo)
                exact = sec * sample_rate + frac_frames
                frames = int(exact)
                frac_frames = exact - frames
                if frames:
                    write_audio(frames)

            if msg.is_meta:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                continue

            ch = getattr(msg, 'channel', 0)
            if msg.type == 'note_on':
                if msg.velocity == 0:
                    lib.fluid_synth_noteoff(synth, ch, msg.note)
                else:
                    lib.fluid_synth_noteon(synth, ch, msg.note, msg.velocity)
            elif msg.type == 'note_off':
                lib.fluid_synth_noteoff(synth, ch, msg.note)
            elif msg.type == 'control_change':
                lib.fluid_synth_cc(synth, ch, msg.control, msg.value)
            elif msg.type == 'program_change':
                lib.fluid_synth_program_change(synth, ch, msg.program)
            elif msg.type == 'pitchwheel':
                lib.fluid_synth_pitch_bend(synth, ch, msg.pitch + 8192)
            elif msg.type == 'aftertouch':
                lib.fluid_synth_channel_pressure(synth, ch, msg.value)
            elif msg.type == 'polytouch':
                lib.fluid_synth_key_pressure(synth, ch, msg.note, msg.value)
            # SysEx is ignored here; GM files usually render correctly without it.

        write_audio(int(round(tail_seconds * sample_rate)))
        ff.stdin.close()
        rc = ff.wait()
        ff = None
        if rc != 0:
            raise RuntimeError(f'ffmpeg exited with code {rc}')

        return rendered_frames / sample_rate
    finally:
        if ff is not None:
            try:
                if ff.stdin:
                    ff.stdin.close()
            except Exception:
                pass
            try:
                ff.terminate()
            except Exception:
                pass
        if synth:
            lib.delete_fluid_synth(synth)
        lib.delete_fluid_settings(settings)


def main():
    ap = argparse.ArgumentParser(description='Offline MIDI -> MP3 using libFluidSynth from Python.')
    ap.add_argument('midi')
    ap.add_argument('mp3')
    ap.add_argument('--soundfont', default='/usr/share/sounds/sf2/TimGM6mb.sf2')
    ap.add_argument('--sample-rate', type=int, default=44100)
    ap.add_argument('--gain', type=float, default=0.55)
    ap.add_argument('--mp3-quality', type=int, default=2)
    ap.add_argument('--tail', type=float, default=4.0)
    args = ap.parse_args()
    dur = render(args.midi, args.mp3, args.soundfont, args.sample_rate,
                 args.gain, args.mp3_quality, args.tail)
    print(f'written: {args.mp3}')
    print(f'audio duration: {dur:.3f} s')


if __name__ == '__main__':
    main()
