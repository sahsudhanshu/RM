"""Assemble Presentation.tex: preamble + extras + body; order the bibliography by FIRST CITATION (IEEE) and compile."""
import os, re, subprocess, sys
ENV = dict(os.environ, TEXMFHOME=os.path.abspath(".texlocal/usr/share/texlive/texmf-dist"))
keys = {1: "base", 2: "maini", 3: "mrs", 4: "fan", 5: "shi", 6: "lin", 7: "icra", 8: "kool", 9: "williams", 10: "hurwitz", 11: "gazebo", 12: "code"}
body = open("body.tex").read()
if re.search(r"(?<![\\a-zA-Z])\[(\d{1,2})\](?!\{)", body):                   # first run: literal [n] -> \cite
    body = re.sub(r"(?<![\\a-zA-Z])\[(\d{1,2})\]", lambda m: "\\cite{%s}" % keys[int(m.group(1))], body)
    body = body.replace("~\\cite", "~\\cite")
    open("body.tex", "w").write(body)
m = re.search(r"\\begin\{thebibliography\}\{99\}(.*?)\\end\{thebibliography\}", body, re.S)
items = dict(re.findall(r"\\bibitem\{(\w+)\}(.*?)(?=\\bibitem|\Z)", m.group(1), re.S))
main = body[:m.start()]
order = []
for k in re.findall(r"\\cite\{(\w+)\}", main):
    if k not in order:
        order.append(k)
missing = [k for k in items if k not in order]
print("first-cite order:", order, "| uncited:", missing)
bib = "\\begin{thebibliography}{99}\n" + "".join("\\bibitem{%s}%s" % (k, items[k].rstrip() + "\n") for k in order) + "\\end{thebibliography}"
body = body[:m.start()] + bib + body[m.end():]
open("Presentation.tex", "w").write(open("preamble.tex").read() + open("extras.tex").read() + body)
for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "Presentation.tex"], capture_output=True, text=True, env=ENV)
    if r.returncode:
        print(r.stdout[-2500:]); sys.exit(1)
print("compiled OK")
