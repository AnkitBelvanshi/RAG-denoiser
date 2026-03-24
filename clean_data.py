import json, os, re, unicodedata

inp = 'outputs/corpora/squad_chunks_noisy.jsonl'
out = 'outputs/corpora/squad_chunks_noisy_norm.jsonl'

# Ensure output directory exists
os.makedirs(os.path.dirname(out), exist_ok=True)

# Set up text cleaning tools
trans = str.maketrans({
    '\u2018': "'", '\u2019': "'", '\u201C': '"', '\u201D': '"',
    '\u2013': '-', '\u2014': '-', '\u00A0': ' ', '\u200B': '', '\uFEFF': ''
})
ws = re.compile(r'[ \t\r\f\v]+')
nl = re.compile(r'\n{3,}')

n = 0
with open(inp, 'r', encoding='utf-8') as f, open(out, 'w', encoding='utf-8') as g:
    for line in f:
        if not line.strip(): 
            continue
        
        r = json.loads(line)
        t = r.get('text', '')
        
        # Normalize and clean text
        t = unicodedata.normalize('NFKC', t).translate(trans)
        t = nl.sub('\n\n', t)
        t = '\n'.join(ws.sub(' ', L).strip() for L in t.split('\n'))
        
        r['text'] = t.strip()
        g.write(json.dumps(r, ensure_ascii=False) + '\n')
        n += 1

print('Wrote', n, '->', out)