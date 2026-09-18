"""
A tiny document DSL rendered to BOTH .docx (python-docx) and .pdf (reportlab),
so the Word and PDF hand-outs can never drift apart.

A document is a list of blocks:
    ('title', text, subtitle)
    ('h1'|'h2'|'h3', text)
    ('p', text)                     inline markup: **bold**, *italic*, `code`
    ('bullets', [text, ...])
    ('numbers', [text, ...])
    ('table', [headers], [[row], ...], [col_widths] | None)
    ('callout', kind, heading, text)     kind: note | warn | tip
    ('code', text)
    ('math', text)                  centred, monospaced
    ('spacer', points)
    ('pagebreak',)
"""
import os
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as RLTTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)


def _register_fonts():
    """reportlab's built-in Helvetica cannot render ₹, ≫, ᵀ or most of the Greek
    this document needs. DejaVu ships with matplotlib, covers every glyph we use,
    and embeds cleanly, so register it and use it throughout."""
    try:
        import matplotlib
        base = os.path.join(os.path.dirname(matplotlib.__file__),
                            'mpl-data', 'fonts', 'ttf')
        faces = [('Body', 'DejaVuSans.ttf'), ('Body-Bold', 'DejaVuSans-Bold.ttf'),
                 ('Body-Italic', 'DejaVuSans-Oblique.ttf'),
                 ('Body-BoldItalic', 'DejaVuSans-BoldOblique.ttf'),
                 ('Mono', 'DejaVuSansMono.ttf'), ('Mono-Bold', 'DejaVuSansMono-Bold.ttf')]
        for name, fn in faces:
            path = os.path.join(base, fn)
            if not os.path.exists(path):
                return False
            pdfmetrics.registerFont(RLTTFont(name, path))
        pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold',
                                      italic='Body-Italic', boldItalic='Body-BoldItalic')
        pdfmetrics.registerFontFamily('Mono', normal='Mono', bold='Mono-Bold',
                                      italic='Mono', boldItalic='Mono-Bold')
        return True
    except Exception:
        return False


HAVE_UNICODE_FONT = _register_fonts()
FONT = 'Body' if HAVE_UNICODE_FONT else 'Helvetica'
FONT_B = 'Body-Bold' if HAVE_UNICODE_FONT else 'Helvetica-Bold'
FONT_M = 'Mono' if HAVE_UNICODE_FONT else 'Courier'

# Fallbacks used only if the Unicode font could not be registered.
_ASCII = {'₹': 'Rs.', '≫': '>>', 'ᵀ': 'T', '‖': '||', '⁻¹': '^-1', '²': '^2',
          '₁': '1', '₂': '2', 'ⱼ': 'j', 'ᵢ': 'i', '·': '.', '×': 'x', '→': '->'}

ACCENT = RGBColor(0x1F, 0x6F, 0x5C)
INK = RGBColor(0x1B, 0x1D, 0x1C)
MUTED = RGBColor(0x4C, 0x51, 0x4F)
RL_ACCENT = colors.HexColor('#1f6f5c')
RL_INK = colors.HexColor('#1b1d1c')
RL_MUTED = colors.HexColor('#4c514f')
RL_LINE = colors.HexColor('#dfe0da')
KIND = {
    'note': ('#e4ecf9', '#2b5fa8'),
    'warn': ('#fae7ea', '#b03a52'),
    'tip':  ('#fbf0dc', '#b4700c'),
    'ok':   ('#e3f1ec', '#1f6f5c'),
}

TOKEN = re.compile(r'(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)')


def _split(text):
    """Split inline markup into (text, style) pairs."""
    out = []
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            out.append((part[2:-2], 'b'))
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            out.append((part[1:-1], 'i'))
        elif part.startswith('`') and part.endswith('`'):
            out.append((part[1:-1], 'c'))
        else:
            out.append((part, ''))
    return out


def _rl(text):
    """Convert inline markup to reportlab mini-HTML."""
    if not HAVE_UNICODE_FONT:
        for k, v in _ASCII.items():
            text = text.replace(k, v)
    text = (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'`([^`]+?)`',
                  rf'<font face="{FONT_M}" size="8.5" color="#2a4d43">\1</font>', text)
    return text


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
def _shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:fill'), hexcolor)
    tcPr.append(shd)


def _runs(par, text, size=10.5, color=INK):
    for chunk, style in _split(text):
        r = par.add_run(chunk)
        r.font.size = Pt(size)
        r.font.color.rgb = color
        if style == 'b':
            r.bold = True
        elif style == 'i':
            r.italic = True
        elif style == 'c':
            r.font.name = 'Consolas'
            r.font.size = Pt(size - 1)
            r.font.color.rgb = RGBColor(0x2A, 0x4D, 0x43)
    return par


def to_docx(blocks, path, footer_text=''):
    doc = Document()
    st = doc.styles['Normal']
    st.font.name = 'Calibri'
    st.font.size = Pt(10.5)
    st.paragraph_format.space_after = Pt(7)
    st.paragraph_format.line_spacing = 1.18
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.85)
        s.left_margin = s.right_margin = Inches(0.9)
        if footer_text:
            p = s.footer.paragraphs[0]
            p.text = footer_text
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.size = Pt(8)
                r.font.color.rgb = MUTED

    for b in blocks:
        kind = b[0]
        if kind == 'title':
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(b[1])
            r.font.size = Pt(26)
            r.bold = True
            r.font.color.rgb = INK
            p.paragraph_format.space_after = Pt(4)
            if len(b) > 2 and b[2]:
                p2 = doc.add_paragraph()
                r2 = p2.add_run(b[2])
                r2.font.size = Pt(12)
                r2.font.color.rgb = MUTED
                p2.paragraph_format.space_after = Pt(16)
        elif kind in ('h1', 'h2', 'h3'):
            size = {'h1': 17, 'h2': 13.5, 'h3': 11.5}[kind]
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16 if kind != 'h3' else 11)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.keep_with_next = True
            r = p.add_run(b[1])
            r.font.size = Pt(size)
            r.bold = True
            r.font.color.rgb = ACCENT if kind == 'h1' else INK
        elif kind == 'p':
            _runs(doc.add_paragraph(), b[1])
        elif kind == 'bullets':
            for item in b[1]:
                p = doc.add_paragraph(style='List Bullet')
                p.paragraph_format.space_after = Pt(3)
                _runs(p, item)
        elif kind == 'numbers':
            for item in b[1]:
                p = doc.add_paragraph(style='List Number')
                p.paragraph_format.space_after = Pt(3)
                _runs(p, item)
        elif kind == 'table':
            headers, rows = b[1], b[2]
            t = doc.add_table(rows=1, cols=len(headers))
            t.style = 'Table Grid'
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for i, h in enumerate(headers):
                c = t.rows[0].cells[i]
                c.text = ''
                _shade(c, 'EEEEEA')
                _runs(c.paragraphs[0], f'**{h}**', size=9.5)
            for row in rows:
                cells = t.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = ''
                    _runs(cells[i].paragraphs[0], str(v), size=9.5)
            doc.add_paragraph().paragraph_format.space_after = Pt(4)
        elif kind == 'callout':
            _, k, head, text = b
            fill = KIND[k][0].lstrip('#')
            t = doc.add_table(rows=1, cols=1)
            t.style = 'Table Grid'
            c = t.rows[0].cells[0]
            _shade(c, fill)
            c.text = ''
            if head:
                ph = c.paragraphs[0]
                rh = ph.add_run(head.upper())
                rh.bold = True
                rh.font.size = Pt(8.5)
                rh.font.color.rgb = RGBColor.from_string(KIND[k][1].lstrip('#'))
                ph.paragraph_format.space_after = Pt(2)
                _runs(c.add_paragraph(), text, size=10)
            else:
                _runs(c.paragraphs[0], text, size=10)
            doc.add_paragraph().paragraph_format.space_after = Pt(4)
        elif kind == 'code':
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.22)
            p.paragraph_format.space_before = Pt(4)
            for i, ln in enumerate(b[1].split('\n')):
                if i:
                    p.add_run('\n')
                r = p.add_run(ln)
                r.font.name = 'Consolas'
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(0x2A, 0x4D, 0x43)
        elif kind == 'math':
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(8)
            r = p.add_run(b[1])
            r.font.name = 'Cambria Math'
            r.font.size = Pt(12)
            r.italic = True
        elif kind == 'spacer':
            doc.add_paragraph().paragraph_format.space_after = Pt(b[1])
        elif kind == 'pagebreak':
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.save(path)
    return path


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def to_pdf(blocks, path, title='', footer_text=''):
    ss = getSampleStyleSheet()
    S = {
        'title': ParagraphStyle('t', parent=ss['Title'], fontName=FONT_B,
                                fontSize=25, leading=29, textColor=RL_INK,
                                alignment=0, spaceAfter=3),
        'sub': ParagraphStyle('sub', parent=ss['Normal'], fontName=FONT, fontSize=11.5, leading=15,
                              textColor=RL_MUTED, spaceAfter=16),
        'h1': ParagraphStyle('h1', parent=ss['Heading1'], fontName=FONT_B,
                             fontSize=16, leading=20, textColor=RL_ACCENT,
                             spaceBefore=16, spaceAfter=6),
        'h2': ParagraphStyle('h2', parent=ss['Heading2'], fontName=FONT_B,
                             fontSize=12.5, leading=16, textColor=RL_INK,
                             spaceBefore=13, spaceAfter=5),
        'h3': ParagraphStyle('h3', parent=ss['Heading3'], fontName=FONT_B,
                             fontSize=10.8, leading=14, textColor=RL_INK,
                             spaceBefore=10, spaceAfter=4),
        'p': ParagraphStyle('p', parent=ss['BodyText'], fontName=FONT, fontSize=9.6, leading=13.6,
                            textColor=RL_INK, alignment=TA_JUSTIFY, spaceAfter=6),
        'li': ParagraphStyle('li', parent=ss['BodyText'], fontName=FONT, fontSize=9.6, leading=13.4,
                             textColor=RL_INK, leftIndent=14, bulletIndent=3, spaceAfter=3),
        'cell': ParagraphStyle('cell', parent=ss['BodyText'], fontName=FONT, fontSize=8.4, leading=11.2,
                               textColor=RL_INK, spaceAfter=0),
        'cellh': ParagraphStyle('ch', parent=ss['BodyText'], fontSize=8.2, leading=11,
                                textColor=RL_MUTED, fontName=FONT_B, spaceAfter=0),
        'code': ParagraphStyle('code', parent=ss['BodyText'], fontName=FONT_M,
                               fontSize=8.4, leading=11.4, textColor=colors.HexColor('#2a4d43'),
                               leftIndent=10, spaceBefore=3, spaceAfter=7,
                               backColor=colors.HexColor('#f1f1ec'),
                               borderPadding=(5, 5, 5, 5)),
        'math': ParagraphStyle('math', parent=ss['BodyText'], fontName=FONT, fontSize=11, leading=16,
                               alignment=TA_CENTER, textColor=RL_INK,
                               spaceBefore=4, spaceAfter=8),
        'callh': ParagraphStyle('calh', parent=ss['BodyText'], fontSize=7.8, leading=10,
                                fontName=FONT_B, spaceAfter=2),
        'callp': ParagraphStyle('calp', parent=ss['BodyText'], fontName=FONT, fontSize=9.3, leading=12.8,
                                textColor=RL_INK, spaceAfter=0),
    }

    story = []
    for b in blocks:
        kind = b[0]
        if kind == 'title':
            story.append(Paragraph(_rl(b[1]), S['title']))
            if len(b) > 2 and b[2]:
                story.append(Paragraph(_rl(b[2]), S['sub']))
        elif kind in ('h1', 'h2', 'h3'):
            story.append(Paragraph(_rl(b[1]), S[kind]))
        elif kind == 'p':
            story.append(Paragraph(_rl(b[1]), S['p']))
        elif kind == 'bullets':
            for item in b[1]:
                story.append(Paragraph(_rl(item), S['li'], bulletText='•'))
            story.append(Spacer(1, 4))
        elif kind == 'numbers':
            for i, item in enumerate(b[1], 1):
                story.append(Paragraph(_rl(item), S['li'], bulletText=f'{i}.'))
            story.append(Spacer(1, 4))
        elif kind == 'table':
            headers, rows = b[1], b[2]
            widths = b[3] if len(b) > 3 and b[3] else None
            data = [[Paragraph(_rl(str(h)), S['cellh']) for h in headers]]
            data += [[Paragraph(_rl(str(v)), S['cell']) for v in row] for row in rows]
            avail = A4[0] - 40 * mm
            cw = [avail * w for w in widths] if widths else [avail / len(headers)] * len(headers)
            t = Table(data, colWidths=cw, repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eeeeea')),
                ('GRID', (0, 0), (-1, -1), 0.5, RL_LINE),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, colors.HexColor('#fafaf7')]),
            ]))
            story.append(t)
            story.append(Spacer(1, 9))
        elif kind == 'callout':
            _, k, head, text = b
            bg, fg = KIND[k]
            inner = []
            if head:
                hs = ParagraphStyle('h', parent=S['callh'], textColor=colors.HexColor(fg))
                inner.append(Paragraph(head.upper(), hs))
            inner.append(Paragraph(_rl(text), S['callp']))
            t = Table([[inner]], colWidths=[A4[0] - 40 * mm], hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg)),
                ('LINEBEFORE', (0, 0), (0, -1), 2.2, colors.HexColor(fg)),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ]))
            story.append(KeepTogether(t))
            story.append(Spacer(1, 9))
        elif kind == 'code':
            esc = b[1].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(esc.replace('\n', '<br/>'), S['code']))
        elif kind == 'math':
            story.append(Paragraph(_rl(b[1]), S['math']))
        elif kind == 'spacer':
            story.append(Spacer(1, b[1]))
        elif kind == 'pagebreak':
            story.append(PageBreak())

    def page(canv, docu):
        canv.saveState()
        canv.setFont(FONT, 7.5)
        canv.setFillColor(RL_MUTED)
        if footer_text:
            canv.drawString(20 * mm, 12 * mm, footer_text)
        canv.drawRightString(A4[0] - 20 * mm, 12 * mm, f'{docu.page}')
        canv.setStrokeColor(RL_LINE)
        canv.setLineWidth(0.4)
        canv.line(20 * mm, 15.5 * mm, A4[0] - 20 * mm, 15.5 * mm)
        canv.restoreState()

    dt = BaseDocTemplate(path, pagesize=A4, title=title,
                         leftMargin=20 * mm, rightMargin=20 * mm,
                         topMargin=18 * mm, bottomMargin=20 * mm)
    frame = Frame(dt.leftMargin, dt.bottomMargin, dt.width, dt.height, id='f')
    dt.addPageTemplates([PageTemplate(id='all', frames=[frame], onPage=page)])
    dt.build(story)
    return path
