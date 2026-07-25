import re

with open('ann.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Fix switchTab
# The old code had: b.getAttribute('onclick')&&b.getAttribute('onclick').includes(`'${group}' `)
# We want to replace it with: includes(`'${group}'`)
html = re.sub(r"includes\(`'\$\{group\}' `\)", r"includes(`'${group}'`)", html)

# 2. Fix markDone to toggle Done / Unmark Done
new_mark_done = """
        function markDone() {
            const cur=sections.find(s=>document.getElementById(s).classList.contains('vis'));
            if( !cur)return;
            
            const nb=document.getElementById('nb-' +cur);
            const btn=document.querySelector('.pill-volt');
            
            if (done.has(cur)) {
                done.delete(cur);
                if(nb)nb.classList.remove('done');
                if(btn) btn.innerHTML = '✓ Mark Done';
            } else {
                done.add(cur);
                if(nb)nb.classList.add('done');
                if(btn) btn.innerHTML = '✕ Unmark Done';
            }
            
            const pct=Math.round(done.size/sections.length*100);
            document.getElementById('pct').textContent=pct+'%';
            document.getElementById('pbar').style.width=pct+'%';
        }
"""
# Replace the original function markDone block
html = re.sub(r'function markDone\(\)\s*\{[^\}]+\}', new_mark_done.strip(), html)

# 3. We also need to update the button text when switching sections so it shows correctly "Mark Done" or "Unmark Done".
# Let's find function show(id) and inject code to update the button text
new_show_inject = """
            if(id==='s6') {
                buildMentorQs(); setTimeout(()=>switchTab('mq', 0), 50);
            }
            const btn=document.querySelector('.pill-volt');
            if(btn) {
                if(done.has(id)) {
                    btn.innerHTML = '✕ Unmark Done';
                } else {
                    btn.innerHTML = '✓ Mark Done';
                }
            }
"""
html = html.replace("if(id==='s6') {\n                buildMentorQs(); setTimeout(()=>switchTab('mq', 0), 50);\n            }", new_show_inject)

with open('ann.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Fixed tabs and mark done.")
