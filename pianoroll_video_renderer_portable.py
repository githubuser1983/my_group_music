import os, json, math, subprocess
from bisect import bisect_right
from PIL import Image, ImageDraw, ImageFont
import mido
import numpy as np

W, H = 1280, 720
FPS = 10
TOP, BOTTOM, LEFT, RIGHT = 80, 50, 80, 40
ROLL_W = W - LEFT - RIGHT
ROLL_H = H - TOP - BOTTOM
WINDOW_SEC = 12.0
PAST_SEC = 2.5
PX_PER_SEC = ROLL_W / WINDOW_SEC
PLAYHEAD_X = LEFT + PAST_SEC * PX_PER_SEC
BG = (10, 12, 18)
ROLL_BG = (18, 22, 30)
GRID = (48, 55, 70)
GRID2 = (28, 34, 44)
TEXT = (235, 240, 248)
SUBTEXT = (175, 185, 200)
PLAYHEAD = (255, 230, 90)
COLORS = [(86,180,233), (230,159,0), (0,158,115), (204,121,167), (240,228,66)]

FONT = ImageFont.load_default()
try:
    FONT_BIG = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 30)
    FONT_MED = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    FONT_SMALL = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
except Exception:
    FONT_BIG = FONT_MED = FONT_SMALL = FONT


def ffprobe_duration(path):
    out = subprocess.check_output([
        'ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',path
    ], text=True).strip()
    return float(out)


def build_tempo_map(mid):
    abs_tick = 0
    tempo = 500000
    events = [(0, tempo)]
    for msg in mid.merged_track:
        abs_tick += msg.time
        if msg.type == 'set_tempo':
            if events and events[-1][0] == abs_tick:
                events[-1] = (abs_tick, msg.tempo)
            else:
                events.append((abs_tick, msg.tempo))
    ticks = [t for t,_ in events]
    tempos = [tp for _,tp in events]
    secs = [0.0]
    for i in range(1, len(events)):
        dt = ticks[i] - ticks[i-1]
        secs.append(secs[-1] + mido.tick2second(dt, mid.ticks_per_beat, tempos[i-1]))
    return ticks, tempos, secs


def tick_to_sec(tick, ticks_per_beat, ticks, tempos, secs):
    i = bisect_right(ticks, tick) - 1
    if i < 0:
        i = 0
    return secs[i] + mido.tick2second(tick - ticks[i], ticks_per_beat, tempos[i])


def parse_notes(mid_path):
    mid = mido.MidiFile(mid_path)
    ticks, tempos, secs = build_tempo_map(mid)
    notes = []
    for ti, track in enumerate(mid.tracks):
        abs_tick = 0
        active = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                key = (getattr(msg,'channel',0), msg.note)
                active.setdefault(key, []).append((abs_tick, msg.velocity))
            elif msg.type in ('note_off','note_on') and (msg.type=='note_off' or msg.velocity==0):
                key = (getattr(msg,'channel',0), msg.note)
                if key in active and active[key]:
                    start_tick, vel = active[key].pop(0)
                    start = tick_to_sec(start_tick, mid.ticks_per_beat, ticks, tempos, secs)
                    end = tick_to_sec(abs_tick, mid.ticks_per_beat, ticks, tempos, secs)
                    if end <= start:
                        end = start + 0.05
                    notes.append({
                        'start': start,
                        'end': end,
                        'pitch': msg.note,
                        'vel': vel,
                        'track': ti,
                        'channel': getattr(msg,'channel',0),
                    })
    notes.sort(key=lambda n: (n['start'], n['pitch']))
    if not notes:
        raise ValueError(f'No notes parsed from {mid_path}')
    return notes


def note_name(n):
    names=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return f"{names[n%12]}{n//12-1}"


def make_background(pmin, pmax, title, subtitle):
    img = Image.new('RGB',(W,H),BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([LEFT, TOP, W-RIGHT, H-BOTTOM], radius=18, fill=ROLL_BG, outline=(36,42,54), width=2)

    # Horizontal pitch stripes and labels
    n_pitches = pmax - pmin + 1
    y_for = lambda p: TOP + (pmax - p + 1) * ROLL_H / n_pitches
    for p in range(pmin, pmax+1):
        y = y_for(p)
        col = GRID if p % 12 == 0 else GRID2
        d.line([LEFT, y, W-RIGHT, y], fill=col, width=1)
        if p % 12 == 0:
            d.text((12, y-8), note_name(p), font=FONT_SMALL, fill=SUBTEXT)

    # Vertical second lines
    for s in np.arange(-PAST_SEC, WINDOW_SEC-PAST_SEC+1e-9, 1.0):
        x = PLAYHEAD_X + s * PX_PER_SEC
        col = GRID if abs(s - round(s)) < 1e-9 else GRID2
        d.line([x, TOP, x, H-BOTTOM], fill=col, width=1)
    # playhead
    d.line([PLAYHEAD_X, TOP, PLAYHEAD_X, H-BOTTOM], fill=PLAYHEAD, width=3)

    d.text((LEFT, 22), title, font=FONT_BIG, fill=TEXT)
    d.text((LEFT, 52), subtitle, font=FONT_MED, fill=SUBTEXT)
    d.text((W-230, 24), 'MIDI-driven piano roll', font=FONT_MED, fill=SUBTEXT)
    return img


def render_pianoroll(midi_path, mp3_path, out_mp4, group_label, meta=None):
    notes = parse_notes(midi_path)
    audio_dur = ffprobe_duration(mp3_path)
    piece_dur = max(max(n['end'] for n in notes), audio_dur)
    pmin = min(n['pitch'] for n in notes)
    pmax = max(n['pitch'] for n in notes)
    # Padding and clamp range reasonably
    pmin = max(21, pmin - 2)
    pmax = min(108, pmax + 2)
    n_pitches = pmax - pmin + 1
    y_for = lambda p: TOP + (pmax - p + 1) * ROLL_H / n_pitches
    bg = make_background(pmin, pmax, f'Motif First — {group_label}', 'order 8 • nearly dry ambient • 3 voices • d minor')
    phrase_info = None
    if meta and os.path.exists(meta):
        try:
            phrase_info = json.load(open(meta))
        except Exception:
            phrase_info = None

    ff = subprocess.Popen([
        'ffmpeg','-y',
        '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
        '-s',f'{W}x{H}','-r',str(FPS),'-i','-',
        '-i', mp3_path,
        '-c:v','libx264','-preset','veryfast','-crf','20',
        '-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart', out_mp4
    ], stdin=subprocess.PIPE)

    total_frames = int(math.ceil(piece_dur * FPS))
    for fi in range(total_frames):
        t = fi / FPS
        frame = bg.copy()
        d = ImageDraw.Draw(frame, 'RGBA')

        # live labels
        mins = int(t // 60)
        secs_now = int(t % 60)
        d.text((W-150, 52), f'{mins:02d}:{secs_now:02d}', font=FONT_MED, fill=TEXT)
        if phrase_info:
            ph = min(int(t // (piece_dur / max(1, phrase_info.get('number_of_phrases', 32)))) + 1, phrase_info.get('number_of_phrases', 32))
            d.text((W-300, H-38), f'phrase {ph}/{phrase_info.get("number_of_phrases",32)}', font=FONT_SMALL, fill=SUBTEXT)

        for n in notes:
            if n['end'] < t - PAST_SEC or n['start'] > t + (WINDOW_SEC - PAST_SEC):
                continue
            x1 = PLAYHEAD_X + (n['start'] - t) * PX_PER_SEC
            x2 = PLAYHEAD_X + (n['end'] - t) * PX_PER_SEC
            if x2 < LEFT or x1 > W-RIGHT:
                continue
            y1 = y_for(n['pitch'])
            y2 = y_for(n['pitch']-1)
            base = COLORS[n['track'] % len(COLORS)]
            alpha = 170 if n['start'] <= t <= n['end'] else 110
            fill = (*base, alpha)
            outline = tuple(min(255, c+35) for c in base) + (230,)
            d.rounded_rectangle([x1, y1+1, x2, y2-1], radius=3, fill=fill, outline=outline, width=1)
            if x1 <= PLAYHEAD_X <= x2:
                d.rounded_rectangle([max(x1, PLAYHEAD_X-2), y1+1, min(x2, PLAYHEAD_X+2), y2-1], radius=2, fill=(255,255,255,150))

        # Footer legend
        leg_y = H-32
        labels = ['voice 1','voice 2','voice 3']
        x = LEFT
        for i, lab in enumerate(labels):
            c = COLORS[i]
            d.rounded_rectangle([x, leg_y, x+18, leg_y+12], radius=3, fill=(*c,220))
            d.text((x+24, leg_y-2), lab, font=FONT_SMALL, fill=SUBTEXT)
            x += 100

        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    rc = ff.wait()
    if rc != 0:
        raise RuntimeError(f'ffmpeg failed for {out_mp4} with code {rc}')


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Render a MIDI-synchronized scrolling piano roll with MP3 audio to MP4.'
    )
    parser.add_argument('--midi', help='Input MIDI for a single video')
    parser.add_argument('--audio', help='Input MP3 for a single video')
    parser.add_argument('--output', help='Output MP4 for a single video')
    parser.add_argument('--group-name', default='Group Music', help='Title for a single video')
    parser.add_argument('--metadata', help='Optional structure JSON for phrase labels')
    parser.add_argument('--data-dir', default=str(Path(__file__).resolve().parent / 'motif_order8_outputs'),
                        help='Folder containing the five MIDI/MP3 pairs')
    parser.add_argument('--group', default='all',
                        choices=['all','Q8','D4','C8','C4xC2','C2xC2xC2'],
                        help='Select one group or render all five')
    args = parser.parse_args()

    if any([args.midi, args.audio, args.output]):
        if not all([args.midi, args.audio, args.output]):
            parser.error('--midi, --audio and --output must be supplied together')
        for path in (args.midi, args.audio):
            if not Path(path).is_file():
                parser.error(f'File not found: {path}')
        render_pianoroll(args.midi, args.audio, args.output, args.group_name, args.metadata)
        print(f'Written: {args.output}')
    else:
        base = Path(args.data_dir)
        groups = ['Q8','D4','C8','C4xC2','C2xC2xC2'] if args.group == 'all' else [args.group]
        for g in groups:
            midi = base / f'motif_first_{g}.mid'
            audio = base / f'motif_first_{g}_nearly_dry_ambient.mp3'
            out = base / f'motif_first_{g}_pianoroll.mp4'
            meta = base / f'motif_first_{g}_structure.json'
            for path in (midi, audio):
                if not path.is_file():
                    parser.error(f'File not found: {path}')
            print(f'Rendering {g} ...', flush=True)
            render_pianoroll(str(midi), str(audio), str(out), g,
                             str(meta) if meta.exists() else None)
            print(f'Written: {out}', flush=True)
