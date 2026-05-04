"""
Throwaway diagnostic — Session 12.
Read what the real Apple 10-K file ACTUALLY contains, not what I assume.
"""
from bs4 import BeautifulSoup
from collections import Counter
import re

PATH = "documents/sec_filings/AAPL_FY2023_10-K.html"

print("=" * 78)
print(f"DIAGNOSING: {PATH}")
print("=" * 78)

with open(PATH, encoding="utf-8", errors="replace") as f:
    content = f.read()
print(f"File size: {len(content):,} chars\n")

# ════════════════════════════════════════════════════════════════════
# A. dei:DocumentFiscalYearFocus — every single occurrence
# ════════════════════════════════════════════════════════════════════
print("─" * 78)
print("A. ALL dei:DocumentFiscalYearFocus TAGS IN XML MODE")
print("─" * 78)

soup_xml = BeautifulSoup(content, "lxml-xml")

fy_tags_xml = []
for tag in soup_xml.find_all(["ix:nonNumeric", "nonNumeric"]):
    if (tag.get("name", "") or "").strip() == "dei:DocumentFiscalYearFocus":
        fy_tags_xml.append(tag)

print(f"XML mode found {len(fy_tags_xml)} dei:DocumentFiscalYearFocus tags")
for i, tag in enumerate(fy_tags_xml):
    text       = tag.get_text(strip=True)
    raw_text   = tag.get_text()
    has_text   = bool(text)
    has_kids   = bool(list(tag.children))
    attrs      = dict(tag.attrs)
    print(f"  [{i}] text={text!r:25s}  raw_text_len={len(raw_text)}  "
          f"has_kids={has_kids}")
    print(f"      attrs: {attrs}")
    # Also dump first 200 chars of inner content
    inner = str(tag)[:300].replace("\n", " ")
    print(f"      inner: {inner}")
print()

# ════════════════════════════════════════════════════════════════════
# B. Same dei tag in HTML mode (lxml)
# ════════════════════════════════════════════════════════════════════
print("─" * 78)
print("B. ALL dei:DocumentFiscalYearFocus TAGS IN HTML MODE")
print("─" * 78)

soup_html = BeautifulSoup(content, "lxml")

# In HTML mode, ix:nonNumeric becomes ix:nonnumeric (lowercased) or just stays
# as the namespaced version
candidates = []
for tag_name in ["ix:nonnumeric", "ix:nonNumeric", "nonnumeric", "nonNumeric"]:
    candidates.extend(soup_html.find_all(tag_name))

fy_tags_html = []
for tag in candidates:
    if (tag.get("name", "") or "").strip() == "dei:DocumentFiscalYearFocus":
        fy_tags_html.append(tag)

print(f"HTML mode found {len(fy_tags_html)} dei:DocumentFiscalYearFocus tags")
for i, tag in enumerate(fy_tags_html[:3]):
    text = tag.get_text(strip=True)
    print(f"  [{i}] tag.name={tag.name}  text={text!r}")
print()

# ════════════════════════════════════════════════════════════════════
# C. Tag-name population in BOTH modes (find the truth about <span>)
# ════════════════════════════════════════════════════════════════════
print("─" * 78)
print("C. TAG COUNTS — XML mode vs HTML mode")
print("─" * 78)

def top_tags(soup, label):
    counts = Counter(t.name for t in soup.find_all())
    print(f"\n{label}: {sum(counts.values()):,} total tags")
    print("  Top 15:")
    for tag, n in counts.most_common(15):
        print(f"    {n:6,d}  {tag}")
    return counts

c_xml  = top_tags(soup_xml,  "XML mode (lxml-xml)")
c_html = top_tags(soup_html, "HTML mode (lxml)")

print()
print(f"  span count  — xml={c_xml.get('span', 0):,}  html={c_html.get('span', 0):,}")
print(f"  div count   — xml={c_xml.get('div', 0):,}  html={c_html.get('div', 0):,}")
print(f"  p count     — xml={c_xml.get('p', 0):,}  html={c_html.get('p', 0):,}")
print(f"  td count    — xml={c_xml.get('td', 0):,}  html={c_html.get('td', 0):,}")
print()

# ════════════════════════════════════════════════════════════════════
# D. Styled elements — what _html_extract_styled_headings ACTUALLY sees
# ════════════════════════════════════════════════════════════════════
print("─" * 78)
print("D. STYLED ELEMENTS WITH font-size — XML mode vs HTML mode")
print("─" * 78)

def count_styled(soup, label):
    cands = soup.find_all(["span", "div", "p", "td"])
    print(f"\n{label}: {len(cands):,} candidate elements (span/div/p/td)")
    with_style = [e for e in cands if e.get("style")]
    print(f"  With any style attr:        {len(with_style):,}")
    with_size = [e for e in with_style
                 if re.search(r"font-size\s*:\s*[\d.]+\s*pt",
                              (e.get("style") or "").lower())]
    print(f"  With inline font-size:Npt:  {len(with_size):,}")
    bold_or_big = []
    for e in with_size:
        s = (e.get("style") or "").lower()
        m = re.search(r"font-size\s*:\s*([\d.]+)\s*pt", s)
        if not m:
            continue
        try:
            pt = float(m.group(1))
        except ValueError:
            continue
        if pt < 11.0:
            continue
        wm = re.search(r"font-weight\s*:\s*([\w\d]+)", s)
        is_bold = False
        if wm:
            w = wm.group(1)
            if w in ("bold", "bolder"):
                is_bold = True
            else:
                try:
                    if int(w) >= 600:
                        is_bold = True
                except ValueError:
                    pass
        if is_bold or pt >= 14.0:
            bold_or_big.append(e)
    print(f"  Heading-quality (≥11pt+bold OR ≥14pt): {len(bold_or_big):,}")
    return bold_or_big

xml_h  = count_styled(soup_xml,  "XML mode")
html_h = count_styled(soup_html, "HTML mode")

print()
print("─" * 78)
print("E. SAMPLE 10 HEADING-QUALITY ELEMENTS (the ones we WANT to find)")
print("─" * 78)

# Use whichever mode found more
src = html_h if len(html_h) > len(xml_h) else xml_h
src_label = "HTML" if len(html_h) > len(xml_h) else "XML"
print(f"\nUsing {src_label} mode source ({len(src):,} candidates)\n")

shown = 0
for e in src:
    text = e.get_text(separator=" ", strip=True)
    if not text or len(text) < 3 or len(text) > 200:
        continue
    if len(text.split()) > 25:
        continue
    style = (e.get("style") or "")[:80]
    print(f"  {text[:65]!r:67s}")
    print(f"      style: {style}")
    shown += 1
    if shown >= 10:
        break

print()
print("─" * 78)
print("F. PART I / PART II PRESENCE CHECK")
print("─" * 78)

# Find any element containing "Part I" / "Part II" — what tag is it?
for needle in ["PART I", "Part I", "PART II", "Part II"]:
    matches = soup_html.find_all(string=re.compile(re.escape(needle)))
    print(f"\n  '{needle}' found in {len(matches)} text nodes (HTML mode)")
    for m in matches[:3]:
        parent = m.parent if m.parent else None
        if parent is None:
            continue
        ptag   = parent.name
        pstyle = (parent.get("style") or "")[:60] if parent else ""
        full_text = m.strip()[:60]
        print(f"     parent=<{ptag}>  style={pstyle!r}")
        print(f"     text:   {full_text!r}")

print("\n" + "=" * 78)
print("DIAGNOSTIC COMPLETE — paste this entire output back to me")
print("=" * 78)