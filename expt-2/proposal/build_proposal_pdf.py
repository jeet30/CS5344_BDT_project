import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and render page numbers
    and professional running headers on content pages.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        page_num = self._pageNumber
        # Cover page has page_num == 1
        if page_num == 1:
            # Draw decorative border on cover page
            self.saveState()
            self.setStrokeColor(colors.HexColor('#1a365d'))
            self.setLineWidth(3)
            self.rect(25, 25, A4[0]-50, A4[1]-50)
            self.setStrokeColor(colors.HexColor('#2b6cb0'))
            self.setLineWidth(1)
            self.rect(29, 29, A4[0]-58, A4[1]-58)
            self.restoreState()
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor('#718096'))

        # Running Header (pages 2+)
        header_text = "CS5344 Course Project Proposal  |  Group XX: Tabular Augmentation for Anomaly Detection"
        self.drawString(36, A4[1] - 28, header_text)
        self.setStrokeColor(colors.HexColor('#cbd5e0'))
        self.setLineWidth(0.6)
        self.line(36, A4[1] - 32, A4[0] - 36, A4[1] - 32)

        # Running Footer (pages 2+)
        self.setStrokeColor(colors.HexColor('#cbd5e0'))
        self.setLineWidth(0.6)
        self.line(36, 34, A4[0] - 36, 34)
        
        # Display content page number (excluding cover page)
        # Content pages: page_num 2 to 6 are Content Pages 1 to 5. Page 7 is References.
        content_page_idx = page_num - 1
        if content_page_idx <= 5:
            footer_page_text = f"Content Page {content_page_idx} of 5"
        else:
            footer_page_text = "References"
            
        self.drawString(36, 22, "National University of Singapore  |  School of Computing")
        self.drawRightString(A4[0] - 36, 22, footer_page_text)
        self.restoreState()

print("NumberedCanvas class ready.")
