#!/usr/bin/env python3
# Minimal, document-tailored Markdown -> LaTeX converter for BIO_LEARNED_GRAPH_PROOF.md
# Targets xelatex. Preserves $...$ / $$...$$ math verbatim; handles tables, lists,
# code fences, bold/italic/code, and the small non-ASCII set present in the file.
import re, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "BIO_LEARNED_GRAPH_PROOF.md"
OUT = sys.argv[2] if len(sys.argv) > 2 else "BIO_LEARNED_GRAPH_PROOF.tex"

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage{amsmath,amssymb}
\usepackage[margin=1in]{geometry}
\usepackage{fontspec}
\usepackage{newunicodechar}
\usepackage{array}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\setmainfont{Latin Modern Roman}
\setmonofont{Latin Modern Mono}
\newunicodechar{§}{\S}
\newunicodechar{–}{\textendash}
\newunicodechar{—}{\textemdash}
\newunicodechar{λ}{\ensuremath{\lambda}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{↔}{\ensuremath{\leftrightarrow}}
\newunicodechar{∎}{\ensuremath{\blacksquare}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.6em}
\begin{document}
"""

def esc_text(t):
    # escape LaTeX specials in plain prose (math/code already split out)
    for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
                 ('#', r'\#'), ('{', r'\{'), ('}', r'\}'), ('_', r'\_'),
                 ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')]:
        t = t.replace(a, b)
    return t

def esc_code(t):
    # inside \texttt{}: escape everything that would break
    for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
                 ('#', r'\#'), ('{', r'\{'), ('}', r'\}'), ('_', r'\_'),
                 ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}'),
                 ('$', r'\$')]:
        t = t.replace(a, b)
    return t

def fmt_plain(seg):
    # bold, then italic, on already-escaped text
    seg = esc_text(seg)
    seg = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', seg)
    seg = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\emph{\1}', seg)
    return seg

# split a line into math ($...$), code (`...`), and plain, process each
TOKEN = re.compile(r'(\$[^$]*\$|`[^`]*`)')
def inline(line):
    out = []
    for part in TOKEN.split(line):
        if not part:
            continue
        if part.startswith('$') and part.endswith('$'):
            out.append(part)                      # math verbatim
        elif part.startswith('`') and part.endswith('`'):
            out.append(r'\texttt{' + esc_code(part[1:-1]) + '}')
        else:
            out.append(fmt_plain(part))
    return ''.join(out)

def heading(line):
    if line.startswith('#### '):
        return r'\subsubsection*{' + inline(line[5:]) + '}'
    if line.startswith('### '):
        return r'\subsection*{' + inline(line[4:]) + '}'
    if line.startswith('## '):
        return r'\section*{' + inline(line[3:]) + '}'
    if line.startswith('# '):
        return r'{\LARGE\bfseries ' + inline(line[2:]) + r'}\par\medskip'
    return None

lines = open(SRC, encoding='utf-8').read().split('\n')
body, i, n = [], 0, len(lines)
while i < n:
    line = lines[i]
    st = line.strip()

    # code fence
    if st.startswith('```'):
        buf = []
        i += 1
        while i < n and not lines[i].strip().startswith('```'):
            buf.append(lines[i].replace('λ', 'lambda'))
            i += 1
        i += 1
        body.append(r'\begin{verbatim}')
        body.extend(buf)
        body.append(r'\end{verbatim}')
        continue

    # horizontal rule (exactly ---)
    if st == '---':
        body.append(r'\medskip\hrule\medskip')
        i += 1
        continue

    # display math on its own line: $$ ... $$
    if st.startswith('$$') and st.endswith('$$') and len(st) > 3:
        inner = st[2:-2]
        body.append(r'\[' + inner + r'\]')
        i += 1
        continue

    # table block
    if st.startswith('|') and i + 1 < n and re.match(r'^\s*\|[\s:|-]+\|\s*$', lines[i+1]):
        header = [c.strip() for c in st.strip('|').split('|')]
        ncol = len(header)
        i += 2  # skip header + separator
        rows = []
        while i < n and lines[i].strip().startswith('|'):
            rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
            i += 1
        w = round(0.92 / ncol, 3)
        colspec = '|' + '|'.join([f'p{{{w}\\linewidth}}' for _ in range(ncol)]) + '|'
        body.append(r'\begin{center}\small')
        body.append(r'\begin{tabular}{' + colspec + '}')
        body.append(r'\hline')
        body.append(' & '.join(r'\textbf{' + inline(c) + '}' for c in header) + r' \\ \hline')
        for r in rows:
            r = (r + [''] * ncol)[:ncol]
            body.append(' & '.join(inline(c) for c in r) + r' \\ \hline')
        body.append(r'\end{tabular}')
        body.append(r'\end{center}')
        continue

    # bullet list block
    if st.startswith('* ') or st.startswith('- '):
        body.append(r'\begin{itemize}[leftmargin=1.4em,itemsep=1pt]')
        while i < n and (lines[i].strip().startswith('* ') or lines[i].strip().startswith('- ')):
            item = lines[i].strip()[2:]
            body.append(r'\item ' + inline(item))
            i += 1
        body.append(r'\end{itemize}')
        continue

    # heading
    h = heading(line)
    if h is not None:
        body.append(h)
        i += 1
        continue

    # blank line
    if st == '':
        body.append('')
        i += 1
        continue

    # plain paragraph line
    body.append(inline(line))
    i += 1

open(OUT, 'w', encoding='utf-8').write(PREAMBLE + '\n'.join(body) + '\n\\end{document}\n')
print("wrote", OUT)
