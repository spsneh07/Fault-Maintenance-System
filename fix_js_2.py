import re

with open('ann.html', 'r', encoding='utf-8') as f:
    html = f.read()

# I want to fix the specific innerHTML block that builds the mentor questions
# because newlines inside the template literal ${...} and between text are creating weird output.

old_block = r'''el.innerHTML\+=` <div class="acc" > <div class="acc-h" onclick="toggleAcc\(this\)" > <div class="acc-q" >Q\$\{
                    i\+1
                \}

                \. \$\{
                    qa\[0\]
                \}

                </div> <div class="acc-meta" > <span class="lvl \$\{lvlMap\[qa\[2\]\]\}" >\$\{
                    lvlLabel\[qa\[2\]\]
                \}

                </span> <span class="acc-arr" >▼</span> </div> </div> <div class="acc-body" >\$\{
                    qa\[1\]
                \}

                </div> </div>`;'''

new_block = '''el.innerHTML += `<div class="acc"><div class="acc-h" onclick="toggleAcc(this)"><div class="acc-q">Q${i+1}. ${qa[0]}</div><div class="acc-meta"><span class="lvl ${lvlMap[qa[2]]}">${lvlLabel[qa[2]]}</span><span class="acc-arr">▼</span></div></div><div class="acc-body">${qa[1]}</div></div>`;'''

# But since my regex might not match perfectly because of spaces, let me just find the `el.innerHTML +=` and replace it via regex matching everything up to `</div> </div>`;`
html = re.sub(r'el\.innerHTML\+=[^;]+`;', new_block, html)

with open('ann.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("JS template literals block replaced.")
