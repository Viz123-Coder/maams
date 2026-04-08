# Project01
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  1. GENERATE SAMPLE DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
n = 200

names = [f"Student_{i:03d}" for i in range(1, n + 1)]
genders = np.random.choice(["Male", "Female"], n, p=[0.48, 0.52])
grades  = np.random.choice(["Grade 9", "Grade 10", "Grade 11", "Grade 12"], n)
study_hours = np.round(np.random.uniform(1, 10, n), 1)

math    = np.clip(np.random.normal(72, 15, n) + study_hours * 1.5, 0, 100).astype(int)
science = np.clip(np.random.normal(68, 14, n) + study_hours * 1.2, 0, 100).astype(int)
english = np.clip(np.random.normal(74, 12, n) + study_hours * 1.0, 0, 100).astype(int)
history = np.clip(np.random.normal(70, 13, n) + study_hours * 0.8, 0, 100).astype(int)

attendance = np.clip(np.random.normal(85, 10, n), 50, 100).astype(int)

df = pd.DataFrame({
    "Name"         : names,
    "Gender"       : genders,
    "Grade"        : grades,
    "Study_Hours"  : study_hours,
    "Math"         : math,
    "Science"      : science,
    "English"      : english,
    "History"      : history,
    "Attendance"   : attendance,
})

df["Average_Score"] = df[["Math", "Science", "English", "History"]].mean(axis=1).round(2)

def assign_letter(score):
    if score >= 90: return "A"
    elif score >= 80: return "B"
    elif score >= 70: return "C"
    elif score >= 60: return "D"
    else: return "F"

df["Letter_Grade"] = df["Average_Score"].apply(assign_letter)

# ─────────────────────────────────────────────
#  2. ANALYTICS
# ─────────────────────────────────────────────
subject_means = df[["Math", "Science", "English", "History"]].mean().round(2)
grade_avg     = df.groupby("Grade")["Average_Score"].mean().sort_index()
gender_avg    = df.groupby("Gender")["Average_Score"].mean()
letter_counts = df["Letter_Grade"].value_counts().reindex(["A","B","C","D","F"], fill_value=0)
top10         = df.nlargest(10, "Average_Score")[["Name","Average_Score","Letter_Grade"]]

# ─────────────────────────────────────────────
#  3. DASHBOARD LAYOUT
# ─────────────────────────────────────────────
DARK_BG   = "#0F1117"
CARD_BG   = "#1A1D27"
ACCENT1   = "#4F8EF7"   # blue
ACCENT2   = "#A78BFA"   # purple
ACCENT3   = "#34D399"   # green
ACCENT4   = "#F472B6"   # pink
ACCENT5   = "#FBBF24"   # amber
TEXT_MAIN = "#E2E8F0"
TEXT_SUB  = "#94A3B8"
GRID_CLR  = "#2D3748"

SUBJECT_COLORS = [ACCENT1, ACCENT2, ACCENT3, ACCENT5]
GRADE_COLORS   = [ACCENT1, ACCENT3, ACCENT4, ACCENT5]
LETTER_COLORS  = {"A": ACCENT3, "B": ACCENT1, "C": ACCENT5, "D": ACCENT4, "F": "#EF4444"}

plt.rcParams.update({
    "figure.facecolor"  : DARK_BG,
    "axes.facecolor"    : CARD_BG,
    "axes.edgecolor"    : GRID_CLR,
    "axes.labelcolor"   : TEXT_MAIN,
    "axes.titlecolor"   : TEXT_MAIN,
    "xtick.color"       : TEXT_SUB,
    "ytick.color"       : TEXT_SUB,
    "text.color"        : TEXT_MAIN,
    "grid.color"        : GRID_CLR,
    "grid.linewidth"    : 0.6,
    "font.family"       : "DejaVu Sans",
})

fig = plt.figure(figsize=(20, 22), facecolor=DARK_BG)
fig.patch.set_facecolor(DARK_BG)

# Main title
fig.text(0.5, 0.975, "🎓 Student Performance Analysis Dashboard",
         ha="center", va="top", fontsize=24, fontweight="bold",
         color=TEXT_MAIN)
fig.text(0.5, 0.960, f"Dataset: {n} students  |  Subjects: Math, Science, English, History",
         ha="center", va="top", fontsize=11, color=TEXT_SUB)

gs = gridspec.GridSpec(4, 3, figure=fig,
                       top=0.945, bottom=0.04,
                       left=0.05, right=0.97,
                       hspace=0.50, wspace=0.35)

# ── helper ────────────────────────────────────
def style_ax(ax, title, subtitle=""):
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_CLR)
        sp.set_linewidth(0.8)
    ax.set_title(title, fontsize=12, fontweight="bold",
                 color=TEXT_MAIN, pad=8, loc="left")
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.01), xycoords="axes fraction",
                    fontsize=8, color=TEXT_SUB, ha="left")
    ax.tick_params(colors=TEXT_SUB, labelsize=8)
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.6, linestyle="--")

# ─────────────────────────────────────────────
#  PLOT 1 – KPI cards (top row, span all 3 cols)
# ─────────────────────────────────────────────
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.set_facecolor(DARK_BG)
ax_kpi.axis("off")

kpis = [
    ("📊 Total Students",  f"{n}",                          ACCENT1),
    ("📈 Class Average",   f"{df['Average_Score'].mean():.1f}%", ACCENT2),
    ("⏰ Avg Study Hrs",   f"{df['Study_Hours'].mean():.1f} hrs", ACCENT3),
    ("🏆 Top Score",       f"{df['Average_Score'].max():.1f}%", ACCENT5),
    ("✅ Avg Attendance",  f"{df['Attendance'].mean():.1f}%", ACCENT4),
    ("🎯 Pass Rate",       f"{(df['Letter_Grade'] != 'F').mean()*100:.1f}%", "#60A5FA"),
]

card_w, card_h = 0.145, 0.75
xs = np.linspace(0.03, 0.87, len(kpis))

for (label, value, color), x in zip(kpis, xs):
    rect = FancyBboxPatch((x, 0.10), card_w, card_h,
                          boxstyle="round,pad=0.02",
                          linewidth=1.5, edgecolor=color,
                          facecolor=CARD_BG, transform=ax_kpi.transAxes,
                          zorder=2)
    ax_kpi.add_patch(rect)
    ax_kpi.text(x + card_w/2, 0.62, value,
                transform=ax_kpi.transAxes,
                ha="center", va="center",
                fontsize=17, fontweight="bold", color=color, zorder=3)
    ax_kpi.text(x + card_w/2, 0.28, label,
                transform=ax_kpi.transAxes,
                ha="center", va="center",
                fontsize=8, color=TEXT_SUB, zorder=3)

# ─────────────────────────────────────────────
#  PLOT 2 – Subject Average Scores (bar)
# ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
bars = ax2.bar(subject_means.index, subject_means.values,
               color=SUBJECT_COLORS, edgecolor=DARK_BG,
               linewidth=0.8, width=0.55, zorder=3)
for bar, val in zip(bars, subject_means.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}", ha="center", va="bottom",
             fontsize=8, color=TEXT_MAIN, fontweight="bold")
ax2.set_ylim(0, 105)
ax2.set_ylabel("Average Score (%)", color=TEXT_SUB, fontsize=8)
style_ax(ax2, "Subject Averages", "Mean score per subject")

# ─────────────────────────────────────────────
#  PLOT 3 – Grade Distribution (letter grades pie)
# ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
colors_pie = [LETTER_COLORS[l] for l in letter_counts.index]
wedges, texts, autotexts = ax3.pie(
    letter_counts.values,
    labels=letter_counts.index,
    autopct="%1.1f%%",
    colors=colors_pie,
    startangle=90,
    pctdistance=0.78,
    wedgeprops=dict(linewidth=1.5, edgecolor=DARK_BG),
)
for t in texts: t.set_color(TEXT_MAIN); t.set_fontsize(10)
for at in autotexts: at.set_color(DARK_BG); at.set_fontsize(8); at.set_fontweight("bold")
ax3.set_facecolor(CARD_BG)
ax3.set_title("Grade Distribution", fontsize=12, fontweight="bold",
              color=TEXT_MAIN, pad=8, loc="left")

# ─────────────────────────────────────────────
#  PLOT 4 – Avg Score by Grade Level (horizontal bar)
# ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
h_bars = ax4.barh(grade_avg.index, grade_avg.values,
                  color=GRADE_COLORS, edgecolor=DARK_BG,
                  linewidth=0.8, height=0.55, zorder=3)
for bar, val in zip(h_bars, grade_avg.values):
    ax4.text(val + 0.3, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}", va="center", ha="left",
             fontsize=8, color=TEXT_MAIN, fontweight="bold")
ax4.set_xlim(0, 110)
ax4.set_xlabel("Average Score (%)", color=TEXT_SUB, fontsize=8)
ax4.grid(axis="x", color=GRID_CLR, linewidth=0.6, linestyle="--")
ax4.grid(axis="y", visible=False)
style_ax(ax4, "Score by Grade Level", "Average across all subjects")

# ─────────────────────────────────────────────
#  PLOT 5 – Study Hours vs Average Score (scatter)
# ─────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
scatter_colors = [ACCENT3 if g == "Female" else ACCENT1 for g in df["Gender"]]
ax5.scatter(df["Study_Hours"], df["Average_Score"],
            c=scatter_colors, alpha=0.55, s=18, zorder=3)
# Trend line
z = np.polyfit(df["Study_Hours"], df["Average_Score"], 1)
p = np.poly1d(z)
xs_trend = np.linspace(df["Study_Hours"].min(), df["Study_Hours"].max(), 100)
ax5.plot(xs_trend, p(xs_trend), color=ACCENT5, linewidth=2, linestyle="--", zorder=4)

from matplotlib.lines import Line2D
legend_els = [Line2D([0],[0], marker='o', color='w', markerfacecolor=ACCENT1,
                     markersize=7, label='Male'),
              Line2D([0],[0], marker='o', color='w', markerfacecolor=ACCENT3,
                     markersize=7, label='Female'),
              Line2D([0],[0], color=ACCENT5, linewidth=2, linestyle='--', label='Trend')]
ax5.legend(handles=legend_els, fontsize=7, facecolor=CARD_BG,
           edgecolor=GRID_CLR, labelcolor=TEXT_MAIN, loc="upper left")
ax5.set_xlabel("Study Hours / Day", color=TEXT_SUB, fontsize=8)
ax5.set_ylabel("Average Score (%)", color=TEXT_SUB, fontsize=8)
style_ax(ax5, "Study Hours vs Score", "Correlation with trend line")

# ─────────────────────────────────────────────
#  PLOT 6 – Attendance vs Average Score (scatter)
# ─────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
ax6.scatter(df["Attendance"], df["Average_Score"],
            c=ACCENT4, alpha=0.50, s=18, zorder=3)
z2 = np.polyfit(df["Attendance"], df["Average_Score"], 1)
p2 = np.poly1d(z2)
xs2 = np.linspace(df["Attendance"].min(), df["Attendance"].max(), 100)
ax6.plot(xs2, p2(xs2), color=ACCENT5, linewidth=2, linestyle="--", zorder=4)
corr = df["Attendance"].corr(df["Average_Score"])
ax6.text(0.97, 0.07, f"r = {corr:.2f}",
         transform=ax6.transAxes, ha="right", fontsize=9,
         color=ACCENT5, fontweight="bold")
ax6.set_xlabel("Attendance (%)", color=TEXT_SUB, fontsize=8)
ax6.set_ylabel("Average Score (%)", color=TEXT_SUB, fontsize=8)
style_ax(ax6, "Attendance vs Score", "Pearson correlation shown")

# ─────────────────────────────────────────────
#  PLOT 7 – Score Distribution Histogram
# ─────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
ax7.hist(df["Average_Score"], bins=20, color=ACCENT2,
         edgecolor=DARK_BG, linewidth=0.6, zorder=3)
mean_score = df["Average_Score"].mean()
ax7.axvline(mean_score, color=ACCENT5, linewidth=2,
            linestyle="--", zorder=4, label=f"Mean: {mean_score:.1f}")
ax7.legend(fontsize=8, facecolor=CARD_BG,
           edgecolor=GRID_CLR, labelcolor=TEXT_MAIN)
ax7.set_xlabel("Average Score (%)", color=TEXT_SUB, fontsize=8)
ax7.set_ylabel("Number of Students", color=TEXT_SUB, fontsize=8)
style_ax(ax7, "Score Distribution", "Frequency histogram")

# ─────────────────────────────────────────────
#  PLOT 8 – Subject Score Box Plot
# ─────────────────────────────────────────────
ax8 = fig.add_subplot(gs[3, 0:2])
bp_data = [df["Math"], df["Science"], df["English"], df["History"]]
bp = ax8.boxplot(bp_data, patch_artist=True,
                 medianprops=dict(color=DARK_BG, linewidth=2.5),
                 whiskerprops=dict(color=TEXT_SUB),
                 capprops=dict(color=TEXT_SUB),
                 flierprops=dict(marker='o', color=TEXT_SUB,
                                 markerfacecolor=TEXT_SUB, markersize=3))
for patch, color in zip(bp['boxes'], SUBJECT_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax8.set_xticklabels(["Math", "Science", "English", "History"],
                    fontsize=9, color=TEXT_MAIN)
ax8.set_ylabel("Score (%)", color=TEXT_SUB, fontsize=8)
style_ax(ax8, "Subject Score Distribution", "Box plot: median, IQR, outliers")

# ─────────────────────────────────────────────
#  PLOT 9 – Top 10 Students (horizontal bar)
# ─────────────────────────────────────────────
ax9 = fig.add_subplot(gs[3, 2])
top10_sorted = top10.sort_values("Average_Score")
bar_colors9  = [LETTER_COLORS[lg] for lg in top10_sorted["Letter_Grade"]]
hb = ax9.barh(range(len(top10_sorted)), top10_sorted["Average_Score"],
              color=bar_colors9, edgecolor=DARK_BG,
              linewidth=0.6, height=0.6, zorder=3)
ax9.set_yticks(range(len(top10_sorted)))
ax9.set_yticklabels([n.replace("Student_", "S") for n in top10_sorted["Name"]],
                    fontsize=7.5)
ax9.set_xlim(60, 103)
ax9.set_xlabel("Average Score (%)", color=TEXT_SUB, fontsize=8)
for bar, val in zip(hb, top10_sorted["Average_Score"]):
    ax9.text(val + 0.2, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}", va="center", ha="left",
             fontsize=7.5, color=TEXT_MAIN, fontweight="bold")
ax9.grid(axis="x", color=GRID_CLR, linewidth=0.6, linestyle="--")
ax9.grid(axis="y", visible=False)
style_ax(ax9, "Top 10 Students", "By overall average")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
fig.text(0.5, 0.012,
         "Generated with Python  |  pandas · matplotlib  |  Student Performance Analysis Dashboard",
         ha="center", fontsize=8, color=TEXT_SUB)

plt.savefig("/mnt/user-data/outputs/student_performance_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print("✅ Dashboard saved → student_performance_dashboard.png")
plt.show()

# ─────────────────────────────────────────────
#  CONSOLE SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("       STUDENT PERFORMANCE SUMMARY REPORT")
print("="*55)
print(f"  Total Students     : {n}")
print(f"  Class Average      : {df['Average_Score'].mean():.2f}%")
print(f"  Highest Score      : {df['Average_Score'].max():.2f}%")
print(f"  Lowest Score       : {df['Average_Score'].min():.2f}%")
print(f"  Std Deviation      : {df['Average_Score'].std():.2f}")
print(f"  Pass Rate (≥60)    : {(df['Letter_Grade']!='F').mean()*100:.1f}%")
print(f"  Avg Attendance     : {df['Attendance'].mean():.1f}%")
print("\n  Subject Averages:")
for subj, val in subject_means.items():
    print(f"    {subj:<10}: {val:.2f}%")
print("\n  Grade Distribution:")
for letter, count in letter_counts.items():
    bar = "█" * (count // 3)
    print(f"    {letter}  {bar:<20} {count:>3} students")
print("="*55)
