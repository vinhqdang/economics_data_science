"""Build the anonymous single-file LaTeX submission (no author details, no .cls, PNG figures, bibliography
inlined as thebibliography) in paper/submission/ and zip it.  Run after compiling the manuscript once."""
import os, re, subprocess, shutil, zipfile
os.chdir(os.path.dirname(os.path.abspath(__file__)))
M = "gdma_storage_siting_manuscript"
OUT = "submission"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)


def expand(s):
    pat = re.compile(r"\\input\{([^}]+)\}")
    while pat.search(s):
        def sub(m):
            f = m.group(1)
            f = f if f.endswith(".tex") else f + ".tex"
            return open(f).read().rstrip("\n") + "\n"
        s = pat.sub(sub, s)
    return s


s = expand(open(M + ".tex").read())
bbl = open(M + ".bbl").read()
s = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}", lambda m: bbl, s)
figs = sorted(set(re.findall(r"figures/([A-Za-z_]+)\.pdf", s)))
for f in figs:
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", f"figures/{f}.pdf", f"{OUT}/{f}"], check=True)
s = re.sub(r"figures/([A-Za-z_]+)\.pdf", r"\1.png", s)
s = "\n".join(l for l in s.split("\n") if not l.lstrip().startswith("%"))
open(f"{OUT}/{M}.tex", "w").write(s)
with zipfile.ZipFile(f"{M}_anonymous_submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(OUT)):
        z.write(os.path.join(OUT, fn), fn)
print(sorted(os.listdir(OUT)))
