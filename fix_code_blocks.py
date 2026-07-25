import re

with open('ann.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Add CSS for the copy button
css_to_add = """
.code-block { position: relative; }
.copy-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    background: var(--rim3);
    color: var(--fg);
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
    cursor: pointer;
    opacity: 0.8;
    transition: opacity 0.2s;
}
.copy-btn:hover { opacity: 1; }
"""
if '.copy-btn {' not in html:
    html = html.replace('</style>', css_to_add + '\n</style>')

# 2. Add JS for copyCode
js_to_add = """
function copyCode(btn) {
    const pre = btn.parentElement.querySelector('pre');
    if(!pre) return;
    navigator.clipboard.writeText(pre.innerText).then(() => {
        const oldText = btn.innerText;
        btn.innerText = 'Copied!';
        setTimeout(() => btn.innerText = oldText, 2000);
    });
}
"""
if 'function copyCode' not in html:
    html = html.replace('// ═══════════════════════════════════════════\n// INIT', js_to_add + '\n// ═══════════════════════════════════════════\n// INIT')

# 3. Add Copy Button and fix spaces inside <pre> blocks
def fix_pre_block(match):
    pre_content = match.group(1)
    
    # Insert spaces around spans
    pre_content = pre_content.replace('</span><span', '</span> <span')
    pre_content = re.sub(r'</span>([a-zA-Z0-9_])', r'</span> \1', pre_content)
    pre_content = re.sub(r'([a-zA-Z0-9_])<span', r'\1 <span', pre_content)
    
    # Try to add newlines before common python keywords to make it somewhat readable
    keywords = ['import ', 'from ', 'def ', 'class ', 'return ', 'for ', 'if ', 'model.compile', 'model.fit', 'model.summary', 'model.evaluate']
    for kw in keywords:
        # only replace if not already following a newline
        pre_content = re.sub(r'(?<!\n)(<span class="[a-z]+">)?' + kw.replace('.', r'\.'), r'\n\1' + kw, pre_content)
    
    # Newline before comments
    pre_content = pre_content.replace('<span class="c">', '\n<span class="c">')
    
    # Return the reconstructed pre block
    return f'<pre>{pre_content}</pre>'

# Process each pre block
html = re.sub(r'<pre>(.*?)</pre>', fix_pre_block, html, flags=re.DOTALL)

# Add the copy button to code-block divs
if '<button class="copy-btn"' not in html:
    html = re.sub(r'(<div class="code-block"[^>]*>)', r'\1\n<button class="copy-btn" onclick="copyCode(this)">Copy Code</button>', html)

with open('ann.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Fixed code blocks and added copy buttons.")
