"""Generate WiPose scientific/guide SVGs and badges with the standard library.

This draws documented data dependencies, not measured radio propagation or poses.
See docs/PIPELINE.md for the implementation behind each arrow.
New artwork and this generator use LICENSE-branding.
"""
from html import escape
from pathlib import Path
from system_figure import desktop, mobile

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, LINE, SIGNAL = '#20252b', '#58616c', '#abb3bf', '#1558d6'


def begin(width, height, title, description):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10" fill="#1558d6"/></marker></defs>',
            f'<rect width="{width}" height="{height}" fill="#ffffff" stroke="#c9cfd6"/>',
            '<g font-family="Arial, Helvetica, sans-serif">']


def text(parts, x, y, value, size=18, color=INK, weight=400):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(value)}</text>')


def arrow(parts, path, dashed=False):
    dash = ' stroke-dasharray="5 5"' if dashed else ''
    parts.append(f'<path d="{path}" fill="none" stroke="{SIGNAL}" stroke-width="2"{dash} marker-end="url(#arrow)"/>')


def flow_node(p, x, y, w, title, lines=(), h=92, size=18, decision=False):
    """Explicit text wrapping keeps labels readable without a layout service."""
    fill = '#eef4ff' if decision else '#ffffff'
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{SIGNAL if decision else LINE}"/>')
    text(p,x+14,y+28,title,size,INK,700)
    for i,line in enumerate(lines):
        text(p,x+14,y+52+i*20,line,15,MUTED)


def flow_begin(width, height, title, description):
    p=begin(width,height,title,description)
    text(p,24,35,title,23,INK,700)
    return p


def finish(p):
    return '\n'.join(p+['</g></svg>'])+'\n'


def capability_badge(label, value, width, color=SIGNAL):
    """Small local badges link to actual source, not inferred capabilities."""
    split=round(width*.43)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="28" role="img" aria-label="{escape(label+": "+value)}">'
            f'<rect width="{width}" height="28" rx="3" fill="{color}"/>'
            f'<path d="M3 0 H{split} V28 H3 Q0 28 0 25 V3 Q0 0 3 0" fill="#20252b"/>'
            '<g fill="white" font-family="Arial, Helvetica, sans-serif" font-size="11" font-weight="700" text-anchor="middle">'
            f'<text x="{split/2}" y="18">{escape(label)}</text>'
            f'<text x="{split+(width-split)/2}" y="18">{escape(value)}</text></g></svg>\n')


def capture_flow(compact=False):
    description='Three transmitters send ESP-NOW beacons to three receivers. Serial CSI frames enter the host bridge, which saves raw logs and publishes MQTT. Recorded HDF5 sessions undergo clock fitting, epoch separation, amplitude processing, gap masking, gridding and five-sample window construction.'
    if compact:
        p=flow_begin(420,868,'Capture → feature windows',description)
        rows=[('3 ESP32-S3 transmitters',['ESP-NOW beacons']),
              ('3 ESP32-S3 receivers',['CSI frames over serial']),
              ('Host bridge',['Timestamp + frame','Save raw logs; publish MQTT']),
              ('Recorded HDF5 session',['Recorder subscribes to MQTT']),
              ('Clock fitting',['Separate board boot epochs']),
              ('Amplitude + gap mask',['Fill short gaps; retain breaks']),
              ('Corrected 100 Hz grid',['Align the nine radio links']),
              ('Five-sample windows',['Feature tensor + validity'])]
        for i,(title,lines) in enumerate(rows):
            y=64+99*i
            flow_node(p,24,y,372,title,lines,h=82,size=19)
            if i<7: arrow(p,f'M210 {y+83} V{y+97}')
        return finish(p)
    p=flow_begin(1120,470,'Capture → feature windows',description)
    for x,title,lines in [(32,'3 transmitters',['ESP32-S3 / ESP-NOW']),
                          (322,'3 receivers',['CSI frames over serial']),
                          (612,'Host bridge',['Timestamp + frame','Publish MQTT']),
                          (902,'Raw logs',['Preserved capture'])]:
        flow_node(p,x,80,186,title,lines,h=100)
    for left in (218,508,798): arrow(p,f'M{left+2} 130 H{left+102}')
    text(p,228,113,'beacons',14,MUTED)
    text(p,518,113,'CSI',14,MUTED)
    arrow(p,'M705 182 V242 H16 V365 H30')
    text(p,718,218,'MQTT',14,MUTED)
    text(p,32,285,'OFFLINE SAMPLE CONSTRUCTION',13,SIGNAL,700)
    rows=[('HDF5 session',['Recorded link data']),('Clock + epochs',['Fit board → host time']),
          ('Amplitude + mask',['Fill short gaps only']),('100 Hz grid',['Corrected common time']),
          ('Feature windows',['5 samples + validity'])]
    for i,(title,lines) in enumerate(rows):
        x=32+216*i
        flow_node(p,x,320,192,title,lines,h=90,size=17)
        if i<4: arrow(p,f'M{x+194} 365 H{x+214}')
    text(p,32,448,'Default window: 5 samples × 56 carriers × 9 links → float16 (280, 3, 3)',17,MUTED)
    return finish(p)


def training_flow(compact=False):
    description='Synchronized camera video passes through RTMDet and RTMPose to produce 18-joint labels. Recorded CSI becomes aligned windows. Timestamp joining, label QA and presence targets produce pairs for normalization and pose-regression training, which saves weights, configuration and statistics in a checkpoint.'
    if compact:
        p=flow_begin(420,760,'Training → checkpoint',description)
        flow_node(p,24,64,372,'Synchronized camera video',['Training-label source'],h=76,size=19)
        flow_node(p,24,172,372,'RTMDet → RTMPose teacher',['Detect person; label pose'],h=76,size=19)
        arrow(p,'M210 142 V170')
        flow_node(p,24,280,176,'18-joint labels',['Teacher output'],h=76,size=17)
        flow_node(p,220,280,176,'Recorded CSI',['Aligned windows'],h=76,size=17)
        arrow(p,'M210 250 V264 H112 V278')
        flow_node(p,24,400,372,'Join timestamps + label QA',['Build presence targets'],h=76,size=19)
        arrow(p,'M112 358 V382 H210 V398')
        arrow(p,'M308 358 V382 H210')
        flow_node(p,24,512,372,'Fit normalization + pose model',['Train on paired CSI and labels'],h=76,size=18)
        arrow(p,'M210 478 V510')
        flow_node(p,24,624,372,'Saved checkpoint',['Weights, configuration, statistics'],h=86,size=19)
        arrow(p,'M210 590 V622')
        return finish(p)
    p=flow_begin(1120,454,'Training → checkpoint',description)
    nodes=[(32,80,236,'Camera recording',['Synchronized video']),
           (316,80,236,'RTMDet → RTMPose',['Teacher labeling']),
           (600,80,236,'18-joint labels',['Training targets']),
           (884,80,204,'Checkpoint',['Weights + config','Saved statistics']),
           (32,280,236,'Recorded CSI',['Session link data']),
           (316,280,236,'Aligned windows',['Same time base as video']),
           (600,280,236,'Join + label QA',['Presence targets']),
           (884,280,204,'Train regression',['Fit normalization','Learn pose mapping'])]
    for x,y,w,title,lines in nodes: flow_node(p,x,y,w,title,lines,h=100,size=18)
    for x,y,end in [(270,130,314),(554,130,598),(270,330,314),(554,330,598),(838,330,882)]:
        arrow(p,f'M{x} {y} H{end}')
    arrow(p,'M718 182 V278')
    arrow(p,'M986 278 V182')
    text(p,32,430,'The camera supplies training labels. Inference consumes CSI plus the saved checkpoint.',17,MUTED)
    return finish(p)


def inference_flow(compact=False):
    description='Live or recorded CSI enters a ring buffer and validity gate. Valid windows use per-link L2 and stored normalization, then the checkpoint-loaded WiSPPN model, presence gating, smoothing and experimental fall rules. Invalid windows increment the drop count and retain the previous displayed pose and state.'
    if compact:
        p=flow_begin(420,980,'Inference + invalid windows',description)
        for y,title,lines in [(64,'Live or recorded CSI',['Incoming link packets']),
                              (170,'Ring buffer',['Build window + validity']),
                              (276,'Valid window?',['Check missing-data mask'])]:
            flow_node(p,24,y,372,title,lines,h=76,size=19,decision=y==276)
        arrow(p,'M210 142 V168'); arrow(p,'M210 248 V274')
        text(p,88,382,'YES',14,SIGNAL,700); text(p,284,382,'NO',14,SIGNAL,700)
        arrow(p,'M112 354 V398'); arrow(p,'M308 354 V398')
        flow_node(p,24,400,176,'Normalize',['Per-link L2','Checkpoint statistics'],h=108,size=18)
        flow_node(p,220,400,176,'Hold display',['Count one drop','Keep previous','pose and state'],h=108,size=18)
        main=[(560,'WiSPPN + trained weights',['18 joints + confidence']),
              (666,'Presence gate + smoothing',['Filter and stabilize pose']),
              (772,'Experimental fall rules',['Update the rule-based state'])]
        for y,title,lines in main: flow_node(p,24,y,290,title,lines,h=76,size=17)
        arrow(p,'M112 510 V538 H169 V558')
        arrow(p,'M169 638 V664'); arrow(p,'M169 744 V770')
        flow_node(p,24,878,372,'Render pose + state',['May retain the previous estimate'],h=76,size=19)
        arrow(p,'M169 850 V876')
        arrow(p,'M308 510 V538 H408 V916 H398')
        return finish(p)
    p=flow_begin(1000,680,'Inference + invalid windows',description)
    for x,y,title,lines in [(40,80,'Live or recorded CSI',['Incoming link packets']),
                            (375,80,'Ring buffer',['Build window + validity']),
                            (710,80,'Valid window?',['Missing-data gate']),
                            (40,240,'Trained checkpoint',['Weights + saved statistics']),
                            (375,240,'Normalize',['Per-link L2 + stored scale']),
                            (710,240,'WiSPPN model',['Extract 18 joints']),
                            (710,400,'Presence + smoothing',['Gate and stabilize pose']),
                            (375,400,'Experimental fall rules',['Update rule-based state']),
                            (40,400,'Render pose + state',['Current or retained estimate']),
                            (710,560,'Hold previous display',['Count drop; retain pose/state'])]:
        flow_node(p,x,y,250,title,lines,h=90,size=17,decision=title=='Valid window?')
    for x,y,end in [(292,125,373),(627,125,708),(292,285,373),(627,285,708)]: arrow(p,f'M{x} {y} H{end}')
    arrow(p,'M835 172 V208 H500 V238'); text(p,848,202,'yes',14,SIGNAL,700)
    arrow(p,'M835 332 V398'); arrow(p,'M708 445 H627'); arrow(p,'M373 445 H292')
    arrow(p,'M962 125 H980 V605 H962'); text(p,956,206,'no',14,SIGNAL,700)
    arrow(p,'M708 605 H165 V492')
    return finish(p)


def main():
    artworks=[('pipeline-overview.svg',desktop()),('pipeline-overview-mobile.svg',mobile())]
    for name,draw in [('capture',capture_flow),('training',training_flow),('inference',inference_flow)]:
        artworks.extend([(f'pipeline-{name}.svg',draw()),(f'pipeline-{name}-mobile.svg',draw(True))])
    for name,label,value,width,color in [('python','CODE','Python',106,SIGNAL),
                                         ('pytorch','MODEL','PyTorch',116,'#c93f24'),
                                         ('esp32','RADIO','ESP32-S3',126,'#3c4858'),
                                         ('links','LINKS','3 TX × 3 RX',142,SIGNAL),
                                         ('pose','POSE','18 joints / 2D',154,SIGNAL)]:
        artworks.append((f'badge-{name}.svg',capability_badge(label,value,width,color)))
    for name,content in artworks:
        (ROOT/'figures'/name).write_text(content,encoding='utf-8')
        print('figures/'+name)


if __name__=='__main__':
    main()
