from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
import datetime


# ── BRAND COLORS ──────────────────────────────────────────────────────────────
C_BG        = colors.HexColor("#0a0c14")
C_ACCENT    = colors.HexColor("#00c6ff")
C_ACCENT2   = colors.HexColor("#4fffb0")
C_NEG       = colors.HexColor("#ff5c7a")
C_NEU       = colors.HexColor("#a78bfa")
C_WHITE     = colors.white
C_DIMTEXT   = colors.HexColor("#8a8fa8")
C_CARD      = colors.HexColor("#13162a")
C_BORDER    = colors.HexColor("#1e2235")

EMOTION_COLORS = {
    "Happy":    "#4fffb0",
    "Surprise": "#fbbf24",
    "Sad":      "#60a5fa",
    "Fear":     "#fb923c",
    "Angry":    "#ff5c7a",
    "Disgust":  "#a78bfa",
    "Neutral":  "#94a3b8",
}


def make_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles['cover_title'] = ParagraphStyle(
        'cover_title',
        fontName='Helvetica-Bold',
        fontSize=28,
        textColor=C_WHITE,
        alignment=TA_CENTER,
        spaceAfter=8,
        leading=34,
    )
    styles['cover_sub'] = ParagraphStyle(
        'cover_sub',
        fontName='Helvetica',
        fontSize=12,
        textColor=C_DIMTEXT,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    styles['section_heading'] = ParagraphStyle(
        'section_heading',
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=C_ACCENT,
        spaceBefore=18,
        spaceAfter=8,
        leading=18,
    )
    styles['body'] = ParagraphStyle(
        'body',
        fontName='Helvetica',
        fontSize=10,
        textColor=C_WHITE,
        spaceAfter=5,
        leading=16,
    )
    styles['body_dim'] = ParagraphStyle(
        'body_dim',
        fontName='Helvetica',
        fontSize=9,
        textColor=C_DIMTEXT,
        spaceAfter=4,
        leading=14,
    )
    styles['qa_question'] = ParagraphStyle(
        'qa_question',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=C_ACCENT2,
        spaceAfter=3,
        leading=15,
    )
    styles['qa_answer'] = ParagraphStyle(
        'qa_answer',
        fontName='Helvetica',
        fontSize=10,
        textColor=C_WHITE,
        spaceAfter=8,
        leftIndent=14,
        leading=15,
    )
    styles['summary_text'] = ParagraphStyle(
        'summary_text',
        fontName='Helvetica',
        fontSize=10,
        textColor=C_WHITE,
        spaceAfter=6,
        leading=17,
    )
    styles['footer'] = ParagraphStyle(
        'footer',
        fontName='Helvetica',
        fontSize=8,
        textColor=C_DIMTEXT,
        alignment=TA_CENTER,
    )
    styles['badge'] = ParagraphStyle(
        'badge',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=C_ACCENT,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    return styles


def draw_background(canvas_obj, doc):
    """Dark background on every page."""
    canvas_obj.saveState()
    canvas_obj.setFillColor(C_BG)
    canvas_obj.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)

    # Top accent bar
    canvas_obj.setFillColor(C_ACCENT)
    canvas_obj.rect(0, A4[1] - 4, A4[0], 4, fill=1, stroke=0)

    # Bottom footer line
    canvas_obj.setStrokeColor(C_BORDER)
    canvas_obj.setLineWidth(0.5)
    canvas_obj.line(40, 36, A4[0] - 40, 36)

    # Footer text
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(C_DIMTEXT)
    canvas_obj.drawCentredString(
        A4[0] / 2, 22,
        "AI Emotion Detection System  ·  Confidential Session Report"
    )
    canvas_obj.restoreState()


def make_pie_chart(distribution):
    """Generate pie chart PNG and return path."""
    labels = list(distribution.keys())
    values = list(distribution.values())
    colors_list = [EMOTION_COLORS.get(l, "#999999") for l in labels]

    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#0a0c14')
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors_list,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops=dict(width=0.65, edgecolor='#0a0c14', linewidth=2),
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(8)
        at.set_fontweight('bold')

    legend_patches = [
        mpatches.Patch(color=colors_list[i], label=f"{labels[i]} ({values[i]}%)")
        for i in range(len(labels))
    ]
    ax.legend(
        handles=legend_patches,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
        frameon=False,
        labelcolor='white',
    )
    ax.set_facecolor('#0a0c14')
    plt.tight_layout()
    path = "pie_chart_report.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a0c14')
    plt.close()
    return path


def make_bar_chart(distribution):
    """Generate horizontal bar chart PNG and return path."""
    labels = list(distribution.keys())
    values = list(distribution.values())
    colors_list = [EMOTION_COLORS.get(l, "#999999") for l in labels]

    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor='#0a0c14')
    bars = ax.barh(labels, values, color=colors_list, height=0.55,
                   edgecolor='#0a0c14', linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val}%', va='center', ha='left',
                color='white', fontsize=8, fontweight='bold')

    ax.set_facecolor('#0a0c14')
    ax.set_xlim(0, max(values) + 12)
    ax.tick_params(colors='white', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#1e2235')
    ax.spines['left'].set_color('#1e2235')
    ax.xaxis.label.set_color('white')
    ax.set_xlabel('Percentage (%)', color='#8a8fa8', fontsize=8)
    fig.patch.set_facecolor('#0a0c14')
    plt.tight_layout()
    path = "bar_chart_report.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a0c14')
    plt.close()
    return path


def section_line(styles):
    return HRFlowable(
        width="100%", thickness=0.5,
        color=C_BORDER, spaceAfter=10, spaceBefore=4
    )


def generate_pdf():
    with open("session_result.json") as f:
        data = json.load(f)

    distribution = data.get("distribution", {})
    analysis     = data.get("analysis", "N/A")
    timeline     = data.get("timeline", [])
    qa_pairs     = data.get("qa_pairs", [])
    summary_text = data.get("therapist_summary", "")
    detected_emotion = data.get("detected_emotion", max(distribution, key=distribution.get) if distribution else "N/A")

    styles = make_styles()

    # ── Generate charts ───────────────────────────────────────────────────────
    pie_path = make_pie_chart(distribution) if distribution else None
    bar_path = make_bar_chart(distribution) if distribution else None

    session_duration = timeline[-1]["time"] if timeline else 0
    now = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")
    dominant = max(distribution, key=distribution.get) if distribution else "N/A"

    elements = []

    # ── COVER SECTION ─────────────────────────────────────────────────────────
    elements.append(Spacer(1, 30))

    # Title block with background table
    title_data = [[
        Paragraph("🧠  AI Emotion Detection", styles['cover_title']),
    ]]
    title_table = Table(title_data, colWidths=[6.3*inch])
    title_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_CARD),
        ('ROUNDEDCORNERS', [12]),
        ('TOPPADDING',    (0,0), (-1,-1), 24),
        ('BOTTOMPADDING', (0,0), (-1,-1), 24),
        ('LEFTPADDING',   (0,0), (-1,-1), 20),
        ('RIGHTPADDING',  (0,0), (-1,-1), 20),
        ('BOX', (0,0), (-1,-1), 1, C_BORDER),
    ]))
    elements.append(title_table)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Session Analysis Report", styles['cover_sub']))
    elements.append(Paragraph(f"Generated: {now}", styles['cover_sub']))
    elements.append(Spacer(1, 20))

    # ── SESSION SUMMARY STATS TABLE ───────────────────────────────────────────
    elements.append(Paragraph("SESSION OVERVIEW", styles['section_heading']))
    elements.append(section_line(styles))

    stat_data = [
        [
            Paragraph(f"<b><font color='#00c6ff'>{detected_emotion}</font></b><br/><font size='8' color='#8a8fa8'>Detected Emotion</font>", styles['body']),
            Paragraph(f"<b><font color='#4fffb0'>{dominant}</font></b><br/><font size='8' color='#8a8fa8'>Dominant (Session)</font>", styles['body']),
            Paragraph(f"<b><font color='#fbbf24'>{session_duration}s</font></b><br/><font size='8' color='#8a8fa8'>Session Duration</font>", styles['body']),
            Paragraph(f"<b><font color='#a78bfa'>{len(qa_pairs)}/8</font></b><br/><font size='8' color='#8a8fa8'>Questions Answered</font>", styles['body']),
        ]
    ]
    stat_table = Table(stat_data, colWidths=[1.57*inch]*4)
    stat_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), C_CARD),
        ('BOX',           (0,0), (-1,-1), 1, C_BORDER),
        ('INNERGRID',     (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING',    (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('RIGHTPADDING',  (0,0), (-1,-1), 10),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(stat_table)
    elements.append(Spacer(1, 20))

    # ── EMOTION DISTRIBUTION ──────────────────────────────────────────────────
    elements.append(Paragraph("EMOTION DISTRIBUTION", styles['section_heading']))
    elements.append(section_line(styles))

    if pie_path and bar_path:
        from reportlab.platypus import Image as RLImage
        chart_data = [[
            RLImage(pie_path, width=2.9*inch, height=2.1*inch),
            RLImage(bar_path, width=2.9*inch, height=2.1*inch),
        ]]
        chart_table = Table(chart_data, colWidths=[3.15*inch, 3.15*inch])
        chart_table.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), C_CARD),
            ('BOX',           (0,0), (-1,-1), 1, C_BORDER),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING',   (0,0), (-1,-1), 8),
            ('RIGHTPADDING',  (0,0), (-1,-1), 8),
            ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ]))
        elements.append(chart_table)
    elements.append(Spacer(1, 8))

    # Distribution detail rows
    dist_rows = [
        [
            Paragraph(f"<font color='{EMOTION_COLORS.get(em, '#fff')}'><b>{em}</b></font>", styles['body']),
            Paragraph(f"<font color='white'>{val}%</font>", styles['body']),
            # Progress bar using table
            Table([['']], colWidths=[max(0.1, val/100 * 3.5)*inch],
                  style=TableStyle([
                      ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(EMOTION_COLORS.get(em, '#999'))),
                      ('TOPPADDING', (0,0), (-1,-1), 5),
                      ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                  ])),
        ]
        for em, val in distribution.items()
    ]
    dist_table = Table(dist_rows, colWidths=[1.2*inch, 0.7*inch, 4.4*inch])
    dist_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), C_CARD),
        ('ROWBACKGROUNDS',(0,0), (-1,-1), [C_CARD, colors.HexColor("#0f1220")]),
        ('BOX',           (0,0), (-1,-1), 1, C_BORDER),
        ('INNERGRID',     (0,0), (-1,-1), 0.3, C_BORDER),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 12),
        ('RIGHTPADDING',  (0,0), (-1,-1), 8),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(dist_table)
    elements.append(Spacer(1, 20))

    # ── BEHAVIOUR ANALYSIS ────────────────────────────────────────────────────
    elements.append(Paragraph("BEHAVIOUR ANALYSIS", styles['section_heading']))
    elements.append(section_line(styles))

    analysis_data = [[Paragraph(analysis, styles['body'])]]
    analysis_table = Table(analysis_data, colWidths=[6.3*inch])
    analysis_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), C_CARD),
        ('BOX',           (0,0), (-1,-1), 1, C_BORDER),
        ('LEFTBORDER',    (0,0), (0,-1), 4, C_ACCENT2),
        ('TOPPADDING',    (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING',   (0,0), (-1,-1), 16),
        ('RIGHTPADDING',  (0,0), (-1,-1), 16),
    ]))
    elements.append(analysis_table)
    elements.append(Spacer(1, 20))

    # ── Q&A SESSION ───────────────────────────────────────────────────────────
    if qa_pairs:
        elements.append(Paragraph("ASSISTANT Q&A SESSION", styles['section_heading']))
        elements.append(section_line(styles))

        for i, pair in enumerate(qa_pairs):
            q_text = pair.get("question", f"Question {i+1}")
            a_text = pair.get("answer", "Skipped")

            a_color = "#8a8fa8" if a_text == "Skipped" else "white"
            a_prefix = "—" if a_text == "Skipped" else "▶"

            row_data = [[
                Paragraph(
                    f"<font color='#8a8fa8' size='8'>Q{i+1}</font>  "
                    f"<font color='#4fffb0'><b>{q_text}</b></font><br/>"
                    f"<font color='#8a8fa8' size='8'>{a_prefix} </font>"
                    f"<font color='{a_color}'>{a_text}</font>",
                    styles['body']
                )
            ]]
            row_table = Table(row_data, colWidths=[6.3*inch])
            bg = C_CARD if i % 2 == 0 else colors.HexColor("#0f1220")
            row_table.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,-1), bg),
                ('BOX',           (0,0), (-1,-1), 0.3, C_BORDER),
                ('TOPPADDING',    (0,0), (-1,-1), 10),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
                ('LEFTPADDING',   (0,0), (-1,-1), 14),
                ('RIGHTPADDING',  (0,0), (-1,-1), 14),
            ]))
            elements.append(row_table)

        elements.append(Spacer(1, 20))

    # ── THERAPIST SUMMARY ─────────────────────────────────────────────────────
    if summary_text:
        elements.append(Paragraph("THERAPIST SUMMARY", styles['section_heading']))
        elements.append(section_line(styles))

        # Split into paragraphs
        paras = [p.strip() for p in summary_text.strip().split('\n') if p.strip()]
        summary_content = [Paragraph(p, styles['summary_text']) for p in paras]
        summary_content_with_space = []
        for p in summary_content:
            summary_content_with_space.append(p)
            summary_content_with_space.append(Spacer(1, 4))

        summary_data = [[summary_content_with_space]]
        summary_table = Table(summary_data, colWidths=[6.3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), C_CARD),
            ('BOX',           (0,0), (-1,-1), 1, C_BORDER),
            ('TOPPADDING',    (0,0), (-1,-1), 16),
            ('BOTTOMPADDING', (0,0), (-1,-1), 16),
            ('LEFTPADDING',   (0,0), (-1,-1), 18),
            ('RIGHTPADDING',  (0,0), (-1,-1), 18),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 20))

    # ── FOOTER ────────────────────────────────────────────────────────────────
    elements.append(Spacer(1, 16))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "This report is generated automatically by the AI Emotion Detection System. "
        "It is intended for personal reflection and should not be used as a clinical diagnosis.",
        styles['footer']
    ))

    # ── BUILD PDF ─────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        "emotion_report.pdf",
        pagesize=A4,
        leftMargin=0.6*inch,
        rightMargin=0.6*inch,
        topMargin=0.7*inch,
        bottomMargin=0.65*inch,
    )
    doc.build(elements, onFirstPage=draw_background, onLaterPages=draw_background)

    # Cleanup temp chart images
    for p in [pie_path, bar_path]:
        if p and os.path.exists(p):
            try: os.remove(p)
            except: pass