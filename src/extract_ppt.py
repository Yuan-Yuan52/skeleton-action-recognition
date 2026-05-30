import zipfile, xml.etree.ElementTree as ET, os

filepath = r'C:\Users\r13941031\Desktop\2026_05_18_meeting.pptx'
out_path = r'D:\r13941031\model\video_action_project\src\ppt_text.txt'

with zipfile.ZipFile(filepath, 'r') as z:
    slides = [f for f in z.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
    slides.sort(key=lambda x: int(x.replace('ppt/slides/slide', '').replace('.xml', '')))
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, s in enumerate(slides):
            f.write(f"\n--- Slide {i+1} ---\n")
            tree = ET.fromstring(z.read(s))
            for node in tree.findall('.//a:t', {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}):
                if node.text:
                    f.write(node.text + "\n")
