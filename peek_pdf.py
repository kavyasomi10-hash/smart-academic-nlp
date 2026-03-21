# peek_pdf.py
import fitz

doc = fitz.open('data/ncert_class8_science.pdf')
print('Total pages:', len(doc))
print('\n--- Page 1 text ---')
print(doc[0].get_text()[:500])
print('\n--- Page 5 text ---')
print(doc[4].get_text()[:500])
print('\n--- Page 10 text ---')
print(doc[9].get_text()[:500])
doc.close()