"""Character accuracy of strategies A/B/C: overall bars + per-subject dot plot."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/swlda_all_studyf_v3'
NAMES = {'A': 'A faults_only', 'B': 'B ptp_150uv', 'C': 'C blink_overlap'}
COLORS = {'A': '#257de0', 'B': '#f47230', 'C': '#00b894'}
MARKERS = {'A': 'o', 'B': 's', 'C': 'D'}
INK, INK2, GRID = '#191919', '#666666', '#e6e6e6'

overall = pd.read_csv(OUT / 'overall_character_summary.csv').set_index('policy')
subj = pd.read_csv(OUT / 'subject_character_summary.csv')

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={'width_ratios': [1, 1.6]})

# Overall accuracy
pols = list(NAMES)
acc = [overall.loc[p, 'character_accuracy'] * 100 for p in pols]
bars = ax1.bar(range(3), acc, width=0.55, color=[COLORS[p] for p in pols], edgecolor='white', linewidth=2)
for i, p in enumerate(pols):
    r = overall.loc[p]
    ax1.text(i, acc[i] + 1.5, f"{acc[i]:.1f}%\n{int(r.correct_characters)}/{int(r.validation_characters)}",
             ha='center', va='bottom', fontsize=9)
    if r.no_output_characters:
        ax1.text(i, 3, f"{int(r.no_output_characters)} no output", ha='center', va='bottom', fontsize=8, color='white')
ax1.set_xticks(range(3), [NAMES[p] for p in pols])
ax1.set_ylim(0, 100)
ax1.set_ylabel('Character accuracy (%)')
ax1.set_title('Overall validation accuracy (180 characters)', loc='left', fontsize=11)

# Per-subject accuracy
order = subj[subj.policy == 'A'].sort_values('character_accuracy').subject_id.tolist()
ypos = {s: i for i, s in enumerate(order)}
for s in order:
    v = subj[subj.subject_id == s].set_index('policy').character_accuracy * 100
    ax2.plot([v.min(), v.max()], [ypos[s]] * 2, color=GRID, lw=2, zorder=1)
for k, p in enumerate(pols):
    d = subj[subj.policy == p]
    ax2.scatter(d.character_accuracy * 100, [ypos[s] + (k - 1) * 0.18 for s in d.subject_id],
                s=42, marker=MARKERS[p], color=COLORS[p], edgecolor='white', linewidth=1.2, zorder=2, label=NAMES[p])
ax2.set_yticks(range(len(order)), order)
ax2.set_xlim(-3, 103)
ax2.set_xlabel('Character accuracy (%), 18 validation characters per subject')
ax2.set_title('Per subject (sorted by strategy A)', loc='left', fontsize=11)
ax2.legend(loc='lower right', frameon=False, fontsize=9)

for ax in (ax1, ax2):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y' if ax is ax1 else 'x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)

fig.suptitle('SWLDA, all StudyF (10 subjects × 3 sessions), within-session 2/3 → 1/3 split',
             x=0.01, ha='left', fontsize=12, fontweight='bold')
fig.tight_layout()
dest = OUT / 'figures'
dest.mkdir(exist_ok=True)
fig.savefig(dest / 'character_accuracy_by_strategy.png', dpi=160)
print(dest / 'character_accuracy_by_strategy.png')
