"""Single source of figure colors: one meaning always maps to one color.

1. Schemes (categorical, Okabe-Ito, colorblind safe): current-only baseline gray,
   main design blue (thickest), S1 orange, S2 bluish green.
2. Ordered k: one purple ramp; k=0 (history only) is black because it is a
   different scheme, not a lighter step.
3. EA conditions: A blue (same as the main design it is), B vermillion,
   C gray dashed.
References are always gray, labelled next to the line, and kept out of legends.
"""
import numpy as np

SCHEMES={
    'current_only':dict(label='Current only',color='#7F7F7F',lw=1.2,marker='o',zorder=3),
    'history_plus_current':dict(label='History S1 + S2 + current (main design)',color='#0072B2',lw=2.8,marker='o',zorder=6),
    'history_se001_plus_current':dict(label='History S1 + current',color='#E69F00',lw=1.8,marker='s',zorder=4),
    'history_se002_plus_current':dict(label='History S2 + current',color='#009E73',lw=1.8,marker='^',zorder=5),
}
SCHEME_REFERENCE=dict(color='#595959',lw=.9)
SCHEME_REFERENCE_STYLES={'history_only':(0,(5,3)),'history_se001_only':(0,(5,2,1,2)),
                         'history_se002_only':(0,(5,2,1,2)),'all_available_reference':(0,(1,2))}

K_COLORS={0:'#000000',3:'#BCBDDC',6:'#807DBA',12:'#54278F'}
K_MARKERS={0:'s',3:'o',6:'^',12:'D'}
K_BAND_COLORS={3:'#807DBA',12:'#54278F'}
K_REFERENCE=dict(color='#A6A6A6',ls=(0,(3,2)),lw=.8)

EA_CONDITIONS={
    'A':dict(label='A: history + current, no alignment',color='#0072B2',ls='-',lw=2.4,marker='o',zorder=5),
    'B':dict(label='B: history + current, session-wise EA',color='#D55E00',ls='-',lw=2.4,marker='D',zorder=6),
    'C':dict(label='C: current only + EA',color='#7F7F7F',ls=(0,(4,2)),lw=1.4,marker='s',zorder=4),
}
EA_REFERENCE=dict(color='#A6A6A6',ls=(0,(1,2)),lw=.9)

BAND_ALPHA=.15
AUC_LIMITS=(.4,1.0)
REFERENCE_TEXT='#595959'


def label_references(ax,items,fontsize=7,color=REFERENCE_TEXT,min_gap=.035):
    """Write reference names at the right edge, nudged apart instead of overlapping.

    Call after the final y-limits are set. ``items`` is a list of (y, text).
    """
    low,high=ax.get_ylim();gap=min_gap*(high-low);last=-np.inf
    for value,text in sorted(items,key=lambda item:item[0]):
        if not np.isfinite(value):continue
        position=max(value,last+gap);last=position
        ax.annotate(text,(1,position),xycoords=('axes fraction','data'),xytext=(3,0),textcoords='offset points',
                    fontsize=fontsize,color=color,va='center',ha='left',annotation_clip=False)
