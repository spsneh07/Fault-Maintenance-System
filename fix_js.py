import re

with open('ann.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Fix spacing inside JS template literals: `$ {` -> `${`
html = re.sub(r'\$\s+\{', '${', html)

# Some newlines inside template literal might also cause issue if they break expressions? 
# Wait, the screenshot shows literal: Q$ { i+1 } . $ { qa[0] }
# My regex `\$\s+\{` will catch `$ {` and replace it with `${`.

with open('ann.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("JS template literals fixed.")
