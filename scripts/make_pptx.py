"""
Generate a 6-slide PowerPoint of the RK-4 unbalance EXPERIMENT DESIGN (no processing):
what was run and why each combination was chosen.

    python -m scripts.make_pptx      # -> RK4_experiments.pptx
"""
from __future__ import annotations

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

from config import ROOT

# palette
INK = RGBColor(0x0F, 0x17, 0x20); MUT = RGBColor(0x5A, 0x66, 0x72)
ACC = RGBColor(0x0F, 0x7F, 0x8F); FAINT = RGBColor(0x8A, 0x95, 0xA1)
D1 = RGBColor(0x2A, 0x78, 0xD6); D2 = RGBColor(0xE2, 0x69, 0x1F)
ST = RGBColor(0x12, 0x87, 0x6A); CP = RGBColor(0x6A, 0x4B, 0xD0)
DANG = RGBColor(0xC0, 0x3A, 0x33)
BG = RGBColor(0xFF, 0xFF, 0xFF); PANEL = RGBColor(0xF2, 0xF5, 0xF7)
WHYBG = RGBColor(0xE5, 0xF1, 0xF2); LINE = RGBColor(0xDC, 0xE3, 0xE9)
SANS, MONO = "Segoe UI", "Consolas"


def slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])   # blank
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    return s


def box(s, l, t, w, h, fill=None, line=None, rounded=True):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                             Inches(l), Inches(t), Inches(w), Inches(h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(1)
    shp.shadow.inherit = False
    return shp


def oval(s, cx, cy, r, fill=None, line=None, lw=2.0):
    shp = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - r), Inches(cy - r),
                             Inches(2 * r), Inches(2 * r))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(lw)
    shp.shadow.inherit = False
    return shp


def tb(s, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    b = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    return tf


def run(p, text, size, color=INK, bold=False, font=SANS, italic=False):
    r = p.add_run(); r.text = text
    f = r.font; f.size = Pt(size); f.color.rgb = color; f.bold = bold
    f.name = font; f.italic = italic
    return r


def line1(tf, *runs, align=PP_ALIGN.LEFT, space_after=4):
    p = tf.paragraphs[0] if not tf.paragraphs[0].runs else tf.add_paragraph()
    p.alignment = align; p.space_after = Pt(space_after)
    for args in runs:
        run(p, *args)
    return p


def header(s, eyebrow, title, tsize=33):
    box(s, 0.62, 0.52, 0.5, 0.07, fill=ACC, rounded=False)
    line1(tb(s, 0.62, 0.62, 12.1, 0.35), (eyebrow.upper(), 12.5, ACC, True, MONO))
    line1(tb(s, 0.6, 0.98, 12.15, 1.0), (title, tsize, INK, True))


def why(s, *runs, t=6.35, label="WHY", accent=ACC, bg=WHYBG):
    box(s, 0.62, t, 12.1, 0.92, fill=bg)
    box(s, 0.62, t, 0.07, 0.92, fill=accent, rounded=False)
    tf = tb(s, 0.85, t + 0.11, 11.7, 0.72)
    line1(tf, (label, 11, accent, True, MONO), space_after=2)
    line1(tf, *runs)


def card(s, l, t, w, h, tag, tagcol, title, desc):
    box(s, l, t, w, h, fill=PANEL, line=LINE)
    tf = tb(s, l + 0.22, t + 0.18, w - 0.44, h - 0.3)
    line1(tf, (tag.upper(), 10.5, tagcol, True, MONO), space_after=3)
    line1(tf, (title, 18, INK, True), space_after=3)
    line1(tf, (desc, 12.5, MUT))


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)

    # ---------- 1 TITLE ----------
    s = slide(prs)
    box(s, 0, 0, 13.333, 0.16, fill=ACC, rounded=False)
    line1(tb(s, 0.62, 1.15, 12, 0.4), ("KFUPM · IMR — ROTORDYNAMICS", 13, ACC, True, MONO))
    line1(tb(s, 0.6, 1.6, 12.1, 1.7),
          ("The Unbalance Experiment Campaign", 46, INK, True))
    line1(tb(s, 0.62, 3.35, 11.2, 1.0),
          ("Bently Nevada RK-4 rotor kit — what we ran, and why each "
           "combination was chosen.", 19, MUT))
    stats = [("95", "test conditions"), ("~560", "acquisitions"),
             ("4", "unbalance configs"), ("2 × 3", "speeds × repeats")]
    for i, (v, l) in enumerate(stats):
        x = 0.62 + i * 3.08
        box(s, x, 5.0, 2.85, 1.5, fill=PANEL, line=LINE)
        tf = tb(s, x + 0.22, 5.2, 2.5, 1.15)
        line1(tf, (v, 34, ACC if i == 0 else INK, True, MONO), space_after=4)
        line1(tf, (l, 13, MUT))

    # ---------- 2 FACTORIAL / KNOBS ----------
    s = slide(prs)
    header(s, "The design — a factorial sweep", "Four knobs, crossed deliberately")
    knobs = [("axis 1 · plane", D1, "Which disk", "disk 1 · disk 2 · both"),
             ("axis 2 · magnitude", D2, "How much", "0.1–2.0 g  (3–60 g·mm)"),
             ("axis 3 · angle", ST, "Which hole", "0–315°, 8 holes @45°"),
             ("axis 4 · speed", CP, "How fast", "sub-critical dwell rpm")]
    for i, (tag, col, ti, de) in enumerate(knobs):
        card(s, 0.62 + i * 3.04, 2.35, 2.82, 1.9, tag, col, ti, de)
    line1(tb(s, 0.62, 4.5, 12, 1.4),
          ("Plus 3 consecutive repeats per setup (measurement noise) and an "
           "analyst-blind hold-out set (unbiased validation).", 15.5, MUT))
    why(s, ("Crossing the axes — not sampling randomly — ", 15, INK),
        ("disentangles severity, phase, plane and speed effects by construction", 15, INK, True),
        (". Any real unbalance is a combination we've already measured.", 15, INK))

    # ---------- 3 FOUR CONFIGS ----------
    s = slide(prs)
    header(s, "Axis 1 · why these four", "The four unbalance configurations")
    cfg = [("config D1", D1, "Disk 1 only", "single-plane, plane 1", "d1"),
           ("config D2", D2, "Disk 2 only", "single-plane, plane 2", "d2"),
           ("in-phase", ST, "Static", "same hole both disks", "in"),
           ("anti-phase", CP, "Couple", "180° opposed", "anti")]
    cw = 2.95
    for i, (tag, col, ti, de, kind) in enumerate(cfg):
        x = 0.62 + i * 3.05
        box(s, x, 2.2, cw, 2.5, fill=PANEL, line=LINE)
        # two disks
        c1x, c2x, cy, r = x + 0.95, x + 2.0, 3.05, 0.42
        oval(s, c1x, cy, r, line=col if kind in ("d1", "in", "anti") else LINE, lw=2.4)
        oval(s, c2x, cy, r, line=col if kind in ("d2", "in", "anti") else LINE, lw=2.4)
        dr = 0.14
        if kind == "d1": oval(s, c1x, cy - r, dr, fill=col)
        elif kind == "d2": oval(s, c2x, cy - r, dr, fill=col)
        elif kind == "in": oval(s, c1x, cy - r, dr, fill=col); oval(s, c2x, cy - r, dr, fill=col)
        else: oval(s, c1x, cy - r, dr, fill=col); oval(s, c2x, cy + r, dr, fill=col)
        tf = tb(s, x + 0.2, 3.75, cw - 0.4, 0.9)
        line1(tf, (tag.upper(), 10.5, col, True, MONO), align=PP_ALIGN.CENTER, space_after=2)
        line1(tf, (ti, 17, INK, True), align=PP_ALIGN.CENTER, space_after=2)
        line1(tf, (de, 12, MUT), align=PP_ALIGN.CENTER)
    why(s, ("Any two-plane unbalance = a ", 15, INK), ("static", 15, ST, True),
        (" (in-phase) + a ", 15, INK), ("couple", 15, CP, True),
        (" (anti-phase) part. Cover both, plus each single plane, and the twin has "
         "seen the complete basis — so it generalizes to any real unbalance.", 15, INK))

    # ---------- 4 MAGNITUDE + ANGLE ----------
    s = slide(prs)
    header(s, "Blocks A · B · C", "Sweeping magnitude, then angle")
    # left: severity ladder
    line1(tb(s, 0.62, 1.95, 6, 0.4), ("BLOCK A · SEVERITY  (32 conditions)", 12.5, D1, True, MONO))
    masses = [("0.1 g", 3), ("0.2 g", 6), ("0.4 g", 12), ("0.8 g", 24),
              ("1.0 g", 30), ("1.2 g", 36), ("1.6 g", 48), ("2.0 g", 60)]
    y0 = 2.45
    for i, (m, g) in enumerate(masses):
        y = y0 + i * 0.42
        line1(tb(s, 0.62, y - 0.03, 0.75, 0.35), (m, 11, MUT, False, MONO), align=PP_ALIGN.RIGHT)
        box(s, 1.5, y, 0.05 + 3.4 * g / 60, 0.26, fill=D1, rounded=False)
        line1(tb(s, 1.5 + 0.15 + 3.4 * g / 60, y - 0.04, 0.7, 0.35),
              (str(g), 11, INK, True, MONO))
    line1(tb(s, 0.62, y0 + 8 * 0.42 + 0.05, 5.5, 0.5),
          ("4 configs × 8 kit masses at 0°. Denser at the low end = detection limit.",
           12.5, MUT))
    # right: angle
    line1(tb(s, 7.0, 1.95, 6, 0.4), ("BLOCKS B + C · PHASE  (48 + 4)", 12.5, D2, True, MONO))
    cx, cy, rr = 8.7, 3.9, 1.15
    oval(s, cx, cy, rr, line=LINE, lw=2)
    oval(s, cx, cy, 0.05, fill=MUT)
    import math
    for i in range(8):
        a = math.radians(-90 + i * 45)
        px, py = cx + rr * math.cos(a), cy + rr * math.sin(a)
        cardinal = i % 2 == 0
        oval(s, px, py, 0.12 if cardinal else 0.08,
             fill=D2 if cardinal else LINE)
    tf = tb(s, 10.4, 2.4, 2.6, 3.2, anchor=MSO_ANCHOR.MIDDLE)
    line1(tf, ("Cardinal 90/180/270°", 13, INK, True), space_after=3)
    line1(tf, ("× 4 masses × 4 configs", 12, MUT), space_after=10)
    line1(tf, ("+ fine 45/135/225/315°", 13, INK, True), space_after=3)
    line1(tf, ("between-hole check", 12, MUT))
    why(s, ("Severity says ", 15, INK), ("how much", 15, INK, True, SANS, True),
        ("; phase says ", 15, INK), ("where", 15, INK, True, SANS, True),
        (". Together they give a correction weight with a size and a direction.", 15, INK))

    # ---------- 5 SPEEDS + RIGOR ----------
    s = slide(prs)
    header(s, "Axis 4 · speed & rigor", "Two sub-critical speeds, blind-checked")
    # speed axis
    ax_y = 2.75
    box(s, 1.0, ax_y, 11.0, 0.04, fill=MUT, rounded=False)
    # resonance band
    box(s, 8.7, ax_y - 0.7, 2.6, 1.4, fill=RGBColor(0xF6, 0xDD, 0xDB), rounded=False)
    line1(tb(s, 8.7, ax_y - 1.05, 2.6, 0.35), ("resonance band", 11, DANG, True, MONO),
          align=PP_ALIGN.CENTER)
    box(s, 9.85, ax_y - 0.7, 0.03, 1.4, fill=DANG, rounded=False)
    line1(tb(s, 8.85, ax_y + 0.75, 2.3, 0.35), ("N_c ≈ 1648 rpm", 12, DANG, True, MONO),
          align=PP_ALIGN.CENTER)
    for cx_, col, lab, frac in [(5.35, D1, "N1 · 1000", "0.61 N_c"),
                                (6.55, ST, "N2 · 1200", "0.73 N_c")]:
        oval(s, cx_, ax_y + 0.02, 0.13, fill=col)
        line1(tb(s, cx_ - 1.0, ax_y - 0.62, 2.0, 0.35), (lab, 13, col, True, MONO),
              align=PP_ALIGN.CENTER)
        line1(tb(s, cx_ - 1.0, ax_y + 0.28, 2.0, 0.3), (frac, 11, FAINT, False, MONO),
              align=PP_ALIGN.CENTER)
    line1(tb(s, 1.0, ax_y + 0.28, 1.2, 0.3), ("0 rpm", 11, FAINT, False, MONO))
    # two rationale boxes
    box(s, 0.62, 4.15, 5.9, 1.9, fill=WHYBG); box(s, 0.62, 4.15, 0.07, 1.9, fill=ACC, rounded=False)
    tf = tb(s, 0.85, 4.3, 5.5, 1.65)
    line1(tf, ("WHY TWO SPEEDS", 11, ACC, True, MONO), space_after=3)
    line1(tf, ("A physical unbalance is speed-independent — estimating it at two "
               "speeds and checking they agree is a built-in internal validation.", 14.5, INK))
    box(s, 6.82, 4.15, 5.9, 1.9, fill=RGBColor(0xF6, 0xDD, 0xDB))
    box(s, 6.82, 4.15, 0.07, 1.9, fill=DANG, rounded=False)
    tf = tb(s, 7.05, 4.3, 5.5, 1.65)
    line1(tf, ("WHY SUB-CRITICAL", 11, DANG, True, MONO), space_after=3)
    line1(tf, ("At Q ≈ 12, taking a heavy mass through the 1648 rpm resonance is "
               "dangerous. Both dwells sit safely below; a live orbit-trip guards every run.",
               14.5, INK))
    why(s, ("Rigor:  ", 15, INK, True), ("3 consecutive repeats per setup", 15, INK),
        (" separate noise from real effects;  a ", 15, INK),
        ("colleague-prepared blind set", 15, INK), (" — including balanced cases — gives an "
         "unbiased accuracy figure and tests false alarms.", 15, INK))

    # ---------- 6 AT A GLANCE ----------
    s = slide(prs)
    header(s, "The campaign at a glance", "What was performed")
    rows = [("Block", "Configurations", "Magnitude", "Angles", "Cond."),
            ("Baseline", "balanced", "0", "—", "1"),
            ("A · Severity", "D1 · D2 · in · anti", "3–60 g·mm  (8)", "0°", "32"),
            ("B · Phase", "D1 · D2 · in · anti", "6–30 g·mm  (4)", "90/180/270°", "48"),
            ("C · Fine phase", "D1", "12 g·mm", "45/135/225/315°", "4"),
            ("Blind hold-out", "loaded + balanced", "sealed", "off-grid", "10")]
    left, top, widths, rh = 0.62, 2.05, [2.5, 3.5, 2.6, 2.9, 0.6], 0.62
    for ri, r in enumerate(rows):
        head = ri == 0
        if not head and ri % 2 == 0:
            box(s, left, top + ri * rh, sum(widths), rh, fill=PANEL, rounded=False)
        x = left
        for ci, cell in enumerate(r):
            tf = tb(s, x + 0.12, top + ri * rh + 0.06, widths[ci] - 0.2, rh - 0.08,
                    anchor=MSO_ANCHOR.MIDDLE)
            col = FAINT if head else INK
            bold = head or ci == 0
            fnt = MONO if (head or ci in (2, 3, 4)) else SANS
            sz = 11 if head else 13.5
            al = PP_ALIGN.RIGHT if ci == 4 else PP_ALIGN.LEFT
            line1(tf, (cell, sz, col, bold, fnt), align=al)
            x += widths[ci]
        if head:
            box(s, left, top + rh - 0.02, sum(widths), 0.02, fill=ACC, rounded=False)
    # totals
    box(s, 0.62, 6.35, 12.1, 0.92, fill=WHYBG); box(s, 0.62, 6.35, 0.07, 0.92, fill=ACC, rounded=False)
    tf = tb(s, 0.85, 6.5, 11.7, 0.7, anchor=MSO_ANCHOR.MIDDLE)
    line1(tf, ("95 conditions · 2 speeds (1000 / 1200 rpm) · 3 repeats each · "
               "~560 acquisitions", 16, INK, True))

    out = ROOT / "RK4_experiments.pptx"
    prs.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
