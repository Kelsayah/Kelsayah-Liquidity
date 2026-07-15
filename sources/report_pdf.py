from datetime import date
from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _clean(value: object) -> str:
    cleaned = str(value).translate(str.maketrans({
        "–": "-", "—": "-", "·": "-", "“": '"', "”": '"',
        "’": "'", "•": "-",
    }))
    return escape(cleaned)


def build_report_pdf(section: str, report: dict) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title=f"Informe - {section}", author="Global Liquidity Monitor",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=22, leading=27, textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER, spaceAfter=7 * mm,
    )
    heading = ParagraphStyle(
        "ReportHeading", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=13, leading=16, textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=4 * mm, spaceAfter=3 * mm,
    )
    body = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=10, leading=15, textColor=colors.HexColor("#1f2937"),
    )
    small = ParagraphStyle(
        "ReportSmall", parent=body, fontSize=8.5, leading=12,
        textColor=colors.HexColor("#64748b"),
    )
    story = [
        Paragraph("GLOBAL LIQUIDITY MONITOR", ParagraphStyle(
            "Kicker", parent=small, alignment=TA_CENTER,
            textColor=colors.HexColor("#2563eb"), fontName="Helvetica-Bold",
        )),
        Spacer(1, 2 * mm),
        Paragraph(_clean(f"Informe - {section}"), title_style),
        Paragraph(_clean(f"Generado el {date.today().strftime('%d/%m/%Y')}"), ParagraphStyle(
            "Date", parent=small, alignment=TA_CENTER,
        )),
        Spacer(1, 7 * mm),
        Paragraph("Situación actual", heading),
        Paragraph(_clean(report["situation"]), body),
        Paragraph("Señales principales", heading),
    ]
    for signal in report["signals"]:
        story.extend([Paragraph(_clean(f"- {signal}"), body), Spacer(1, 1.2 * mm)])

    story.extend([Paragraph("Escenarios más probables", heading), Spacer(1, 1 * mm)])
    table_data = [[
        Paragraph("Escenario", body), Paragraph("Probabilidad", body),
        Paragraph("Qué implicaría", body),
    ]]
    for scenario in report["scenarios"]:
        table_data.append([
            Paragraph(_clean(scenario["Escenario"]), body),
            Paragraph(f'<b>{scenario["Probabilidad"]}%</b>', body),
            Paragraph(_clean(scenario["Qué implicaría"]), body),
        ])
    scenario_table = Table(table_data, colWidths=[48 * mm, 28 * mm, 82 * mm], repeatRows=1)
    scenario_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([
        scenario_table, Spacer(1, 7 * mm),
        Paragraph(
            "Metodología: los porcentajes son estimaciones heurísticas basadas en los "
            "indicadores del dashboard. No son probabilidades estadísticas calibradas ni "
            "constituyen una recomendación de inversión.", small,
        ),
    ])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(18 * mm, 8 * mm, "Global Liquidity Monitor")
        canvas.drawRightString(192 * mm, 8 * mm, f"Página {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
