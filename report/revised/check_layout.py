"""Inspect the compiled revision and render contact sheets for manual review."""
from pathlib import Path
import hashlib
import json
import pymupdf as pdf

HERE = Path(__file__).resolve().parent
doc = pdf.open(HERE / 'report_revised.pdf')
report = {'total_pages': len(doc), 'main_content_pages': len(doc)-2, 'pages': []}
for i, page in enumerate(doc):
    words = page.get_text('words')
    outside = [w[4] for w in words if w[0] < 20 or w[2] > page.rect.width-20 or w[1] < 15 or w[3] > page.rect.height-15]
    report['pages'].append({'pdf_page': i+1, 'words': len(words), 'opening': page.get_text()[:90], 'outside_safe_page_area': outside})
for name in ['report.pdf', 'report.tex']:
    report['original_'+name+'_sha256'] = hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()
(HERE / 'layout_check.json').write_text(json.dumps(report, indent=2))
for start in range(0,len(doc),4):
    sheet = pdf.open()
    canvas = sheet.new_page(width=595*2, height=842*2)
    for slot, p in enumerate(range(start,min(start+4,len(doc)))):
        x, y = slot%2*595, slot//2*842
        canvas.show_pdf_page(pdf.Rect(x,y,x+595,y+842),doc,p)
    canvas.get_pixmap(matrix=pdf.Matrix(0.8,0.8)).save(HERE/f'preview_{start+1:02d}.png')
print(json.dumps(report, indent=2))
