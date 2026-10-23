"""Original scientific Poseborne figure. Standard-library vector source.

The signal texture and BODY-18 glyph are explicitly schematic, not results.
Default shapes come from train/csi_train/model.py and rt/csi_rt/infer.py.
PNG exports preserve the installed Avenir Next / DIN typography without
redistributing font files. New artwork uses LICENSE-branding.
"""
from html import escape
from math import sin

INK='#172231'
MUTED='#506074'
BLUE='#1465c7'
PURPLE='#6d46a5'
GREEN='#137b64'
TRAINING='#b12370'
HAIR='#d6dfe9'


def label(p,x,y,value,size=20,color=INK,weight=500,mono=False,display=False,anchor=None):
    family='Menlo, Consolas, monospace' if mono else ('DIN Alternate, Avenir Next, Segoe UI, sans-serif' if display else 'Avenir Next, Segoe UI, Helvetica, sans-serif')
    align=f' text-anchor="{anchor}"' if anchor else ''
    p.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{color}"{align}>{escape(value)}</text>')


def start(w,h,title):
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">',
       f'<title id="title">{escape(title)}</title>',
       '<desc id="desc">Three transmitters and three receivers provide nine CSI links. The default amplitude feature tensor has shape 280 by 3 by 3. A ResNet-18-style WiSPPN encoder produces a 3 by 18 by 18 pose adjacency matrix; its diagonal gives 18 two-dimensional joints and confidence. A separate synchronized-camera teacher and paired-CSI training branch saves a required, unbundled checkpoint. Signal textures and pose glyphs are schematic.</desc>',
       f'<rect width="{w}" height="{h}" fill="white"/>','<defs>']
    for name,color in [('signal',BLUE),('model',PURPLE),('pose',GREEN),('train',TRAINING)]:
        p.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10" fill="{color}"/></marker>')
    p.append('</defs>')
    return p


def path(p,d,color=BLUE,width=2.2,arrow=True,dashed=False):
    name={BLUE:'signal',PURPLE:'model',GREEN:'pose',TRAINING:'train'}.get(color,'signal')
    marker=f' marker-end="url(#{name})"' if arrow else ''
    dash=' stroke-dasharray="7 6"' if dashed else ''
    p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{marker}{dash}/>')


def board(p,x,y,scale=1):
    p.append(f'<g transform="translate({x} {y}) scale({scale})">')
    p.append(f'<rect width="32" height="48" rx="4" fill="#eaf2fc" stroke="{BLUE}" stroke-width="1.8"/>')
    p.append(f'<rect x="7" y="16" width="18" height="18" rx="2" fill="{BLUE}"/>')
    p.append(f'<path d="M6 10 H11 V5 H17 V10 H25 M6 39 H26" fill="none" stroke="{BLUE}" stroke-width="1.8"/>')
    for i in range(4):
        p.append(f'<path d="M3 {19+i*4} H7 M25 {19+i*4} H29" stroke="{BLUE}"/>')
    p.append('</g>')


# OpenPose BODY-18 ordering, including neck, eyes and ears.
JOINTS=[(.50,.12),(.50,.25),(.35,.27),(.24,.43),(.13,.57),(.65,.27),(.76,.43),(.86,.51),
        (.40,.53),(.36,.73),(.31,.96),(.60,.53),(.64,.73),(.70,.96),(.46,.10),(.54,.10),(.40,.12),(.60,.12)]
EDGES=[(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),
       (1,11),(11,12),(12,13),(0,14),(14,16),(0,15),(15,17)]


def pose(p,x,y,w,h,color=GREEN,joints=True):
    for a,b in EDGES:
        ax,ay=JOINTS[a]; bx,by=JOINTS[b]
        path(p,f'M{x+ax*w:.2f} {y+ay*h:.2f} L{x+bx*w:.2f} {y+by*h:.2f}',color,2.5,False)
    if joints:
        for a,b in JOINTS:
            p.append(f'<circle cx="{x+a*w:.2f}" cy="{y+b*h:.2f}" r="3" fill="white" stroke="{color}" stroke-width="1.8"/>')


def radio_array(p,left,right,ys,board_scale=1,person_x=None,person_y=None,person_w=80,person_h=135):
    startx=left+32*board_scale
    endx=right
    for a in ys:
        for b in ys:
            p.append(f'<path d="M{startx} {a+24*board_scale} L{endx} {b+24*board_scale}" stroke="{BLUE}" stroke-width="1.5" opacity=".22"/>')
    if person_x is not None:
        x,y,w,h=person_x,person_y,person_w,person_h
        p.append(f'<circle cx="{x+w*.5}" cy="{y+h*.1}" r="{h*.07}" fill="white" stroke="#8fa0b4" stroke-width="2.5"/>')
        path(p,f'M{x+w*.5} {y+h*.2} V{y+h*.59} M{x+w*.28} {y+h*.32} H{x+w*.72} M{x+w*.28} {y+h*.32} L{x+w*.1} {y+h*.57} M{x+w*.72} {y+h*.32} L{x+w*.9} {y+h*.57} M{x+w*.5} {y+h*.59} L{x+w*.28} {y+h*.96} M{x+w*.5} {y+h*.59} L{x+w*.72} {y+h*.96}', '#8fa0b4',2.8,False)
    for y in ys: board(p,left,y,board_scale);board(p,right,y,board_scale)


def amplitude(p,x,y,w,h):
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#f1f6fd"/>')
    for k in range(8): path(p,f'M{x+k*w/7} {y} V{y+h}',HAIR,1,False)
    for row in range(5):
        values=[]
        for k in range(57):
            xx=x+k*w/56
            yy=y+(row+.5)*h/5+sin(k*.28+row*.62)*h*.043+sin(k*.77+row*.3)*h*.018
            values.append(f'{xx:.2f} {yy:.2f}')
        path(p,'M'+' L'.join(values),BLUE,1.8,False)


def tensor(p,x,y,cell=21):
    side=cell*3
    for d in (18,12,6):
        p.append(f'<path d="M{x+d} {y-d} H{x+side+d} V{y+side-d} H{x+d} Z" fill="#e7f0fc" stroke="{BLUE}" stroke-width="1.3"/>')
    for r in range(3):
        for c in range(3):
            fills=['#b5d3f6','#6ca7e8','#2679d2','#97c0ee','#d7e7fa']
            p.append(f'<rect x="{x+c*cell}" y="{y+r*cell}" width="{cell}" height="{cell}" fill="{fills[(r*2+c)%5]}" stroke="white" stroke-width="1.5"/>')


def slab(p,x,bottom,h,w=34,depth=12):
    top=bottom-h
    p.append(f'<path d="M{x} {top} l{depth} -{depth} h{w} v{h} l-{depth} {depth} Z" fill="#e7ddf3" stroke="{PURPLE}" stroke-width="1.5"/>')
    p.append(f'<path d="M{x} {top} h{w} v{h} h-{w} Z" fill="#cdb6e6" stroke="{PURPLE}" stroke-width="1.5"/>')
    path(p,f'M{x+w} {top} l{depth} -{depth} M{x+w} {top} V{bottom}',PURPLE,1.2,False)
    for dy in range(16,int(h),16): path(p,f'M{x+4} {top+dy} H{x+w-4}',PURPLE,.7,False)


def encoder(p,xs,bottom,heights,compact=False):
    for i,(x,h) in enumerate(zip(xs,heights)):
        slab(p,x,bottom,h,w=30 if compact else 36,depth=10 if compact else 12)
        if i<3:
            start=x+(43 if compact else 50)
            path(p,f'M{start} {bottom-h/2} L{xs[i+1]-9} {bottom-heights[i+1]/2}',PURPLE,2.2)
        label(p,x+(18 if compact else 22),bottom+28,'150' if i<2 else '300',17 if compact else 19,PURPLE,600,mono=True,anchor='middle')
        label(p,x+(18 if compact else 22),bottom+52,['144²','72²','36²','18²'][i],16 if compact else 18,MUTED,mono=True,anchor='middle')


def pam(p,x,y,size=102):
    for d in (16,8):
        p.append(f'<rect x="{x+d}" y="{y-d}" width="{size}" height="{size}" fill="#e6f2ef" stroke="{GREEN}" stroke-width="1.5"/>')
    for r in range(18):
        for c in range(18):
            color=GREEN if r==c else '#e6f2ef'
            p.append(f'<rect x="{x+c*size/18:.2f}" y="{y+r*size/18:.2f}" width="{size/18:.2f}" height="{size/18:.2f}" fill="{color}" stroke="white" stroke-width=".55"/>')


def camera(p,x,y):
    p.append(f'<rect x="{x}" y="{y}" width="64" height="42" rx="5" fill="#fff0f8" stroke="{TRAINING}" stroke-width="2"/>')
    p.append(f'<circle cx="{x+34}" cy="{y+21}" r="12" fill="white" stroke="{TRAINING}" stroke-width="2"/>')
    p.append(f'<path d="M{x+8} {y} v-7 h18 v7 M{x+25} {y+43} v12 m-13 0 h26" fill="none" stroke="{TRAINING}" stroke-width="2"/>')


def dataset(p,x,y):
    for d in (12,6,0):
        p.append(f'<rect x="{x+d}" y="{y-d}" width="66" height="48" fill="white" stroke="{TRAINING}" stroke-width="1.5"/>')
    for row in range(4): path(p,f'M{x+9} {y+10+row*8} H{x+56}',TRAINING,1.4,False)


def network(p,x,y):
    for a in range(3):
        for b in range(3): path(p,f'M{x} {y+a*18} L{x+48} {y+b*18}',TRAINING,1,False)
    for xx in (x,x+48):
        for i in range(3):p.append(f'<circle cx="{xx}" cy="{y+i*18}" r="5" fill="white" stroke="{TRAINING}" stroke-width="1.8"/>')


def checkpoint(p,x,y):
    p.append(f'<path d="M{x} {y} h35 l15 15 v45 h-50 Z M{x+35} {y} v15 h15" fill="#fff0f8" stroke="{TRAINING}" stroke-width="2"/>')
    for i in range(3): path(p,f'M{x+10} {y+28+i*9} H{x+39}',TRAINING,1.5,False)


def desktop():
    p=start(1440,885,'Poseborne: from radio links to 18-joint 2D pose')
    label(p,40,49,'WiFi → 18-joint 2D pose',36,INK,700,display=True)
    label(p,40,83,'Single-person WiSPPN-ESP · default amplitude / PAM route',21,MUTED)
    for x,color,txt in [(760,BLUE,'CSI input'),(945,PURPLE,'Learned model'),(1175,TRAINING,'Training only')]:
        p.append(f'<circle cx="{x}" cy="43" r="5" fill="{color}"/>');label(p,x+14,49,txt,18,color,600)
    path(p,'M40 109 H1400',HAIR,1,False)
    p.append('<rect x="632" y="124" width="477" height="466" rx="10" fill="#f8f5fc"/>')
    for x,value,color in [(40,'RADIO ARRAY',BLUE),(355,'AMPLITUDE FEATURES',BLUE),(650,'LEARNED POSE MAPPING',PURPLE),(1135,'POSE REPRESENTATION',GREEN)]:
        label(p,x,145,value,18,color,700,display=True)
    label(p,40,178,'3 TX × 3 RX · ESP32-S3',21,INK,600)
    label(p,52,211,'TX',18,BLUE,700);label(p,262,211,'RX',18,BLUE,700)
    radio_array(p,48,260,[228,326,424],person_x=127,person_y=273,person_w=90,person_h=135)
    label(p,40,505,'Nine links → serial CSI',21,BLUE,600)
    label(p,40,536,'Host bridge or',19,MUTED);label(p,40,561,'compatible recorded session',19,MUTED)
    path(p,'M306 348 H341')
    label(p,355,195,'Schematic amplitude',20,MUTED)
    amplitude(p,355,226,235,119)
    label(p,355,378,'5 samples × 56 carriers',20,BLUE,600)
    path(p,'M473 388 V411')
    tensor(p,430,431,22)
    label(p,356,533,'X: 280 × 3 × 3',25,BLUE,700,mono=True)
    label(p,355,565,'Mask / grid → L2 + saved stats',18,MUTED)
    path(p,'M519 465 H626 V350 H653')
    label(p,650,180,'WiSPPN · ResNet-18-style',25,INK,700,display=True)
    label(p,650,213,'Upsample 3×3 → 144×144; stem 280→150',18,MUTED)
    label(p,650,246,'Four residual groups · two blocks each',18,PURPLE,600)
    encoder(p,[665,765,865,965],435,[150,112,80,52])
    label(p,650,527,'channels / spatial resolution',18,MUTED)
    label(p,650,561,'Decode: 300 → 64 → 3 channels',20,PURPLE,600)
    path(p,'M1015 409 H1088 V307 H1162',PURPLE)
    label(p,1135,195,'PAM: 3 × 18 × 18',22,GREEN,700,mono=True)
    pam(p,1170,244,104)
    label(p,1135,382,'Diagonal → (x, y, ĉ)',21,GREEN,600)
    path(p,'M1222 393 V411',GREEN)
    pose(p,1168,409,112,149)
    label(p,1135,586,'18 joints · schematic pose',19,MUTED)
    path(p,'M40 610 H1400',HAIR,1,False)
    label(p,40,650,'TRAINING ONLY',20,TRAINING,700,display=True)
    camera(p,58,696)
    label(p,90,790,'Camera video',19,TRAINING,600,anchor='middle')
    label(p,90,815,'synchronized',17,MUTED,anchor='middle')
    path(p,'M132 719 H215',TRAINING,2.2,True,True)
    pose(p,245,678,70,78,TRAINING)
    label(p,280,790,'RTMDet → RTMPose',19,TRAINING,600,anchor='middle')
    label(p,280,815,'18-joint labels',17,MUTED,anchor='middle')
    path(p,'M335 719 H495',TRAINING,2.2,True,True)
    dataset(p,511,697)
    label(p,550,790,'Timestamp join',19,TRAINING,600,anchor='middle')
    label(p,550,815,'CSI + camera labels',17,MUTED,anchor='middle')
    path(p,'M473 574 V667 H544 V680',TRAINING,2.2,True,True)
    path(p,'M593 719 H753',TRAINING,2.2,True,True)
    network(p,775,697)
    label(p,800,790,'Fit normalization',19,TRAINING,600,anchor='middle')
    label(p,800,815,'Train pose regression',17,MUTED,anchor='middle')
    path(p,'M838 719 H1045',TRAINING,2.2,True,True)
    checkpoint(p,1062,684)
    label(p,1087,790,'Save checkpoint',19,TRAINING,600,anchor='middle')
    label(p,1087,815,'Required; not bundled',17,MUTED,anchor='middle')
    path(p,'M1087 676 V589 H855 V576',TRAINING,2.2,True,True)
    label(p,1115,653,'Load weights +',18,TRAINING,600);label(p,1115,678,'saved normalization',18,TRAINING,600)
    label(p,40,865,'Solid arrows: inference data path. Dashed magenta: training and checkpoint loading. Signal and pose graphics are schematic.',18,MUTED)
    return '\n'.join(p+['</svg>'])+'\n'


def mobile():
    p=start(420,1365,'Poseborne scientific system figure, narrow layout')
    label(p,24,36,'WiFi → 18 joints in 2D',27,INK,700,display=True)
    label(p,24,66,'Single person · default amplitude / PAM',17,MUTED)
    label(p,24,106,'3 TX × 3 RX · ESP32-S3',22,BLUE,700,display=True)
    label(p,42,134,'TX',16,BLUE,700);label(p,335,134,'RX',16,BLUE,700)
    label(p,210,145,'nine CSI links',16,MUTED,anchor='middle')
    radio_array(p,36,334,[151,222,293],.78,person_x=153,person_y=183,person_w=110,person_h=127)
    label(p,24,367,'Amplitude → feature tensor',23,BLUE,700,display=True)
    amplitude(p,24,397,177,73)
    tensor(p,281,408,18)
    path(p,'M210 436 H259')
    label(p,24,493,'Schematic signal',16,MUTED)
    label(p,231,493,'280 × 3 × 3',20,BLUE,700,mono=True)
    label(p,24,525,'5 samples × 56 carriers; mask / grid',18,MUTED)
    label(p,24,551,'Per-link L2 + saved normalization',18,MUTED)
    p.append('<rect x="24" y="581" width="372" height="238" rx="8" fill="#f8f5fc"/>')
    path(p,'M355 435 H394 V572 H14 V700 H32')
    label(p,24,600,'WiSPPN · ResNet-18-style',23,PURPLE,700,display=True)
    label(p,24,628,'3×3 → 144×144; stem 280→150',18,MUTED)
    encoder(p,[38,128,218,308],756,[104,84,61,42],True)
    label(p,24,840,'PAM → diagonal → 2D pose',23,GREEN,700,display=True)
    path(p,'M351 734 H382 V855 H91 V875',PURPLE)
    pam(p,40,892,91)
    label(p,25,1022,'3 × 18 × 18',19,GREEN,700,mono=True)
    path(p,'M150 938 H241',GREEN)
    label(p,166,922,'diag',16,GREEN,600)
    pose(p,257,869,111,143)
    label(p,257,1039,'Schematic pose',16,MUTED)
    label(p,24,1074,'18 × (x, y) + joint confidence',20,GREEN,600)
    path(p,'M24 1096 H396',HAIR,1,False)
    label(p,24,1133,'TRAINING ONLY',19,TRAINING,700,display=True)
    camera(p,27,1165)
    label(p,110,1171,'RTMDet → RTMPose',19,TRAINING,600)
    label(p,110,1198,'Camera teacher labels',18,MUTED)
    label(p,110,1221,'+ synchronized CSI pairs',18,MUTED)
    network(p,33,1250)
    path(p,'M59 1223 V1241',TRAINING,2,True,True)
    label(p,110,1271,'Fit model + stats',19,TRAINING,600)
    checkpoint(p,334,1250)
    path(p,'M282 1274 H323',TRAINING,2,True,True)
    path(p,'M384 1280 H407 V680 H398',TRAINING,2,True,True)
    label(p,231,669,'load weights + stats',16,TRAINING,600)
    label(p,24,1340,'Checkpoint required; not bundled',17,TRAINING,600)
    return '\n'.join(p+['</svg>'])+'\n'
