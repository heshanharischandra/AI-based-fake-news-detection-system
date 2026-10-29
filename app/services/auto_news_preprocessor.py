
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from app.services.ocr_service import available_languages, configure_tesseract, select_ocr_language, tessdata_config
from app.services.language_service import detect_language, normalize_text

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'uploads' / 'processed'

def identify_language(text):
    return detect_language(text)


def is_overlay_text(text):
    text = text.strip().lower()
    return bool(re.search(r'©|\bsubscribe\b|\bwatch on\b|\byoutube\b|\bada\s+derana\b|www\.|https?://|@[a-z\d_.]{3,}', text))


def _extract_lines(im, lang='sin+eng', psm=11, min_conf=25):
    lang = select_ocr_language() or lang
    grayscale = ImageOps.autocontrast(ImageOps.grayscale(im))
    scale = 2 if max(im.size) < 1800 else 1
    work = ImageEnhance.Contrast(grayscale).enhance(1.25)
    if scale > 1:
        work = work.resize((im.width * scale, im.height * scale), Image.Resampling.LANCZOS)
    words = pytesseract.image_to_data(work, lang=lang, config=f'{tessdata_config()} --oem 3 --psm {psm}', output_type=Output.DICT)
    groups = defaultdict(list)
    for i, term in enumerate(words['text']):
        token = normalize_text(term)
        if not token:
            continue
        try:
            c = float(words['conf'][i])
            if c < min_conf:
                continue
            x, y, w, h = [int(words[f][i]) for f in ('left', 'top', 'width', 'height')]
            x1, y1, x2, y2 = max(0,x//scale), max(0,y//scale), min(im.width,(x+w+scale-1)//scale), min(im.height,(y+h+scale-1)//scale)
            if x2 <= x1 or y2 <= y1:
                continue
            key = (words['block_num'][i],words['par_num'][i],words['line_num'][i])
            groups[key].append((x1,y1,x2,y2,token,c))
        except (ValueError,TypeError):
            continue
    lines=[]
    for vals in groups.values():
        vals.sort(key=lambda k:k[0])
        tx=' '.join(v[4] for v in vals)
        bb=[min(v[0] for v in vals),min(v[1] for v in vals),max(v[2] for v in vals),max(v[3] for v in vals)]
        lines.append({'text':tx,'bbox':bb,'confidence':round(sum(v[5] for v in vals)/len(vals),1), 'language':identify_language(tx),'overlay_candidate':is_overlay_text(tx)})
    return sorted(lines,key=lambda q:(q['bbox'][1],q['bbox'][0]))


def _headline_candidates(lines, im):
    candidates=[]
    for l in lines:
        clean=l['text'].strip()
        bb=l['bbox']; h=bb[3]-bb[1]
        letters=sum(c.isalpha() for c in clean)
        if l['overlay_candidate'] or letters < 5 or l['confidence'] < 35 or h < 9:
            continue
        # Font size, OCR quality and moderately long phrases raise ranking;
        # no position is assumed because headlines appear above or below photos.
        score=1.7*min(1.0,h/max(1,im.height)*11) + 0.8*l['confidence']/100 + 0.6*min(1.0,letters/25)
        candidates.append((score,l))
    candidates.sort(key=lambda p:p[0],reverse=True)
    if not candidates: return []
    best=candidates[0][1]
    # A second line must be close, of similar size and roughly horizontally aligned.
    joined=[best]
    for score,l in candidates[1:]:
        if len(joined)>=3: break
        b=best['bbox']; c=l['bbox']
        ht1=b[3]-b[1];ht2=c[3]-c[1]
        vertical_gap=max(0,max(b[1]-c[3],c[1]-b[3]))
        horizontal_overlap=max(0,min(b[2],c[2])-max(b[0],c[0]))
        related = 0.55 <= ht2/max(1,ht1) <= 1.85 and vertical_gap <= 1.2*max(ht1,ht2) and (horizontal_overlap>0 or abs(b[0]-c[0])<0.12*im.width)
        if related:
            joined.append(l)
    return sorted(joined,key=lambda l:(l['bbox'][1],l['bbox'][0]))


def _visual_candidate(im, selected):
    """Conservative cropped photograph band; may contain multiple photo panels."""
    w,h=im.size
    if not selected:
        return (0,0,w,h),'unsegmented_full_image'
    y1=min(l['bbox'][1] for l in selected)
    y2=max(l['bbox'][3] for l in selected)
    # The biggest continuous band OUTSIDE the detected headline is used as a
    # visual candidate. It is NOT a verified photo-only region.
    upper=max(0,y1-6)
    lower=min(h,y2+6)
    top_height=upper
    bottom_height=h-lower
    if top_height>=bottom_height and top_height>=0.27*h:
        return (0,0,w,upper),'heuristic_above_headline'
    if bottom_height>=0.27*h:
        return (0,lower,w,h),'heuristic_below_headline'
    return (0,0,w,h),'unsegmented_full_image'


def analyze_automatically(image_path):
    path=Path(image_path)
    if not path.is_file(): raise FileNotFoundError(path)
    if configure_tesseract() is None:
        raise RuntimeError('Tesseract is not installed or discoverable. Set TESSERACT_CMD or install Tesseract OCR.')
    language = select_ocr_language(available_languages())
    if language is None:
        raise RuntimeError('Tesseract needs eng.traineddata and/or sin.traineddata.')
    with Image.open(path) as opened:
        im=ImageOps.exif_transpose(opened).convert('RGB')
    lines=_extract_lines(im, lang=language)
    selected=_headline_candidates(lines,im)
    chosen='\n'.join(x['text'] for x in selected).strip()
    lang=identify_language(chosen)
    # Quality proxy, not probability of headline correctness.
    avg_conf=sum(x['confidence'] for x in selected)/len(selected) if selected else 0
    letters=sum(c.isalpha() for c in chosen)
    reliable=(bool(chosen) and lang=='en' and letters>=9 and avg_conf>=70 and not is_overlay_text(chosen))
    headline_status='candidate_accepted_for_demo' if reliable else 'needs_review' if chosen else 'no_claim_detected'
    full_text='\n'.join(l['text'] for l in lines)
    coords,method=_visual_candidate(im,selected)
    folder=OUT/f'{path.stem}_{uuid4().hex[:8]}'
    line_dir=folder/'text_regions'
    visual_dir=folder/'visual_regions'
    for d in (line_dir,visual_dir):d.mkdir(parents=True,exist_ok=True)
    preview=im.copy(); painter=ImageDraw.Draw(preview)
    for ix,line in enumerate(lines,1):
        b=line['bbox']; bbox=(max(0,b[0]-4),max(0,b[1]-4),min(im.width,b[2]+4),min(im.height,b[3]+4))
        dst=line_dir/f'line_{ix:03d}_{line["language"]}.png'
        im.crop(bbox).save(dst)
        line['crop_path']=str(dst.resolve())
        painter.rectangle(bbox,outline='red',width=2)
    for line in selected:painter.rectangle(line['bbox'],outline='lime',width=4)
    vis_path=visual_dir/'automatic_candidate.png'
    im.crop(coords).save(vis_path)
    painter.rectangle(coords,outline='blue',width=4)
    preview_path=folder/'preview.png';preview.save(preview_path)
    result={
        'status':'complete', 'original_path':str(path.resolve()),'original_modified':False,
        'ocr_text':full_text, 'language':lang,'text_regions':lines,
        'headline':chosen,'headline_status':headline_status,'headline_confidence_proxy':round(avg_conf,1),
        'visual_path':str(vis_path.resolve()),'visual_region':list(coords),'visual_region_method':method,
        'visual_region_status':'heuristic_not_validated','is_composite_or_overlay_possible':bool(lines),
        'watermark_status':'overlay_text_candidates_only','overlay_text_regions':[l['bbox'] for l in lines if l['overlay_candidate']],
        'image_size':list(im.size), 'preview_path':str(preview_path.resolve()),'output_dir':str(folder.resolve()),
        'notes':['Headline selection is heuristic and can be wrong',
                 'Sinhala OCR from stylized text requires validation; do not infer accuracy from Tesseract confidence',
                 'Photograph region may include collage panels or watermarks',
                 'AI-image classifier results on composites are not trustworthy',
                 'No original pixels are removed or edited'],
    }
    manifest=folder/'analysis.json';result['manifest_path']=str(manifest.resolve())
    manifest.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result
