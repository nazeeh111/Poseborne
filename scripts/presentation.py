"""Generate the original WiPose overview SVGs with only Python's standard library.

This draws documented data dependencies, not measured radio propagation or poses.
See docs/PIPELINE.md for the implementation behind each arrow.
New artwork and this generator use LICENSE-branding.
"""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, LINE, SIGNAL, ACCENT = '#20252b', '#58616c', '#abb3bf', '#1558d6', '#1558d6'


def begin(width, height, title, description):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10" fill="#1558d6"/></marker></defs>',
            f'<rect width="{width}" height="{height}" fill="#ffffff" stroke="#c9cfd6"/>',
            '<g font-family="Arial, Helvetica, sans-serif">']


def text(parts, x, y, value, size=18, color=INK, weight=400):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(value)}</text>')


def card(parts, x, y, w, h, label, title, lines, accent=SIGNAL):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="{LINE}"/>')
    text(parts, x+16, y+26, label, 13, accent, 700)
    text(parts, x+16, y+57, title, 22, INK, 700)
    for i, line in enumerate(lines):
        text(parts, x+16, y+84+i*23, line, 16, MUTED)


def arrow(parts, path, dashed=False):
    dash = ' stroke-dasharray="5 5"' if dashed else ''
    parts.append(f'<path d="{path}" fill="none" stroke="{SIGNAL}" stroke-width="2"{dash} marker-end="url(#arrow)"/>')


def desktop():
    p=begin(1120, 520, 'WiPose: capture, learn, replay',
            'CSI capture flows through alignment and feature windows into a checkpoint-loaded model and 18 two-dimensional joints. Training combines paired CSI with camera teacher labels to create that checkpoint. Hardware and trained accuracy are not validated by this diagram.')
    text(p,32,48,'Implemented data path',28,INK,700)
    text(p,32,83,'Radio samples, a trained checkpoint and 18 joints in 2D',25)
    text(p,32,120,'LIVE OR RECORDED CSI',13,SIGNAL,700)
    cards=[('01 / HARDWARE','Capture',['3 transmitters','3 receivers / 9 links']),
           ('02 / PROCESSING','Align + gate',['Corrected 100 Hz grid','Mask missing samples']),
           ('03 / MODEL INPUT','Build windows',['5 samples × 56 carriers','280 × 3 × 3 features']),
           ('04 / CHECKPOINT','Estimate pose',['Stored normalization','Learned model weights']),
           ('05 / OUTPUT','18 joints in 2D',['Presence + smoothing','Experimental fall state'])]
    for i,(lab,title,lines) in enumerate(cards):
        x=32+216*i
        card(p,x,140,192,124,lab,title,lines)
        if i<4: arrow(p,f'M{x+194} 202 H{x+214}')
    p.append('<path d="M32 313 H1088" stroke="#c9cfd6"/>')
    text(p,32,335,'TRAINING ONLY',13,ACCENT,700)
    card(p,152,344,232,124,'CAMERA → TEACHER','Generate labels',['Synchronized video','18-joint training targets'],ACCENT)
    card(p,448,344,192,124,'CSI + LABELS','Train model',['Paired samples','Pose regression'],ACCENT)
    card(p,680,344,192,124,'SAVED ARTIFACT','Checkpoint',['Weights + statistics','Required for inference'],ACCENT)
    arrow(p,'M386 406 H446',True)
    arrow(p,'M642 406 H678',True)
    arrow(p,'M560 266 V342',True)
    text(p,574,294,'paired CSI',14,MUTED)
    arrow(p,'M776 342 V266',True)
    text(p,790,294,'load',14,MUTED)
    text(p,32,500,'Implemented dependencies, not a hardware or accuracy validation. The signal walkthrough stops at stage 03.',16,MUTED)
    return '\n'.join(p+['</g></svg>'])+'\n'


def mobile():
    p=begin(420, 320, 'WiPose system map, compact view',
            'Radio samples become aligned CSI windows. Pose inference requires a supplied trained checkpoint and produces 18 joints in two dimensions. Camera labels plus paired CSI train that checkpoint separately.')
    text(p,24,32,'INPUT → MODEL → OUTPUT',14,SIGNAL,700)
    rows=[('CSI windows','Live packets or recorded session'),
          ('Checkpoint-loaded model','Trained weights + saved statistics'),
          ('18 joints in 2D','Single-person research pipeline')]
    for i,(title,detail) in enumerate(rows):
        y=50+79*i
        p.append(f'<path d="M24 {y} H396" stroke="{LINE}"/>')
        text(p,24,y+28,title,21,INK,700)
        text(p,24,y+51,detail,16,MUTED)
        if i<2:arrow(p,f'M210 {y+57} V{y+75}')
    text(p,24,304,'Training: camera labels + paired CSI → checkpoint.',14,MUTED)
    return '\n'.join(p+['</g></svg>'])+'\n'


def main():
    for name,content in [('pipeline-overview.svg',desktop()),('pipeline-overview-mobile.svg',mobile())]:
        (ROOT/'figures'/name).write_text(content,encoding='utf-8')
        print('figures/'+name)


if __name__=='__main__':
    main()
