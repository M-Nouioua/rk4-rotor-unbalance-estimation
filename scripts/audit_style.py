"""
Check the manuscript source against the required writing conventions.

Rules enforced: no em dashes, no bold in the main text, no first person, no
contractions, British spelling, at most three references per citation group,
captions at most 30 words. Comment lines are ignored, since they are not
manuscript text.

    python -m scripts.audit_style

Exit codes: 0 clean, 1 at least one violation.
"""
from __future__ import annotations

import pathlib
import re
import sys

PAPER = pathlib.Path(__file__).resolve().parent.parent / "paper"

RULES = {
    "em dash":           r"\u2014|(?<!-)---(?!-)",
    "bold in main text": r"\\textbf\{",
    "first person":      r"\b(?:we|our|ours|us)\b",
    "contraction":       r"\b\w+(?:n't|'re|'ve|'ll|'m)\b",
    "US spelling":       (r"\b(?:behavior|behaviors|modeling|analyze\w*|"
                          r"generalization|localization|parameteriz\w*|"
                          r"randomized|normaliz\w*|characteriz\w*|"
                          r"discretiz\w*|center|centered|optimiz\w*)\b"),
}


def strip_comments(s: str) -> str:
    out = []
    for line in s.split("\n"):
        i = 0
        while True:
            i = line.find("%", i)
            if i == -1:
                break
            if i == 0 or line[i - 1] != "\\":
                line = line[:i]
                break
            i += 1
        out.append(line)
    return "\n".join(out)


# Proper nouns that must keep their own spelling regardless of the manuscript
# convention. An institution's name is not a spelling choice.
PROPER_NOUNS = ("Interdisciplinary Research Center for Intelligent Manufacturing",)


def main() -> int:
    files = {f.name: strip_comments(f.read_text(encoding="utf-8"))
             for f in sorted(PAPER.glob("*.tex"))
             if not f.name.endswith(".unused")}
    # Names wrap across source lines, so match them with flexible whitespace.
    for noun in PROPER_NOUNS:
        pat = re.compile(r"\s+".join(map(re.escape, noun.split())))
        files = {k: pat.sub("", v) for k, v in files.items()}

    # The CRediT statement sets author names in bold, which is the publisher's
    # house convention for that block. The no-bold rule governs body prose, so
    # the block is excluded rather than the convention broken.
    credit = re.compile(r"\\section\*\{CRediT.*?(?=\\section\*)", re.S)
    files = {k: credit.sub("", v) for k, v in files.items()}

    # The title is set bold at the authors' request. It is not body prose, so it
    # is excluded for the same reason as the CRediT block.
    title = re.compile(r"\\title\{.*?\n?.*?\}", re.S)
    files = {k: title.sub("", v, count=1) for k, v in files.items()}
    body = "\n".join(files.values())
    bad = 0

    print("style rules:")
    for name, pat in RULES.items():
        hits = [(f, m.group(0), s[max(0, m.start() - 45):m.end() + 25].replace("\n", " "))
                for f, s in files.items() for m in re.finditer(pat, s, re.I)]
        print(f"  {'ok ' if not hits else 'FAIL'}  {name:18s} {len(hits)}")
        for f, tok, ctx in hits[:4]:
            print(f"          {f}: ...{ctx.strip()}...")
        bad += len(hits)

    groups = [g for g in re.findall(r"\\cite\{([^}]*)\}", body) if g.count(",") >= 3]
    print(f"  {'ok ' if not groups else 'FAIL'}  citation groups >3   {len(groups)}")
    for g in groups[:4]:
        print(f"          {g}")
    bad += len(groups)

    caps = re.findall(r"\\caption\{(.*?)\}\s*\n\s*\\label", body, re.S)
    over = []
    for c in caps:
        words = len(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", c).split())
        if words > 30:
            over.append((words, " ".join(c.split())[:56]))
    print(f"  {'ok ' if not over else 'FAIL'}  captions >30 words   "
          f"{len(over)} of {len(caps)}")
    for w, t in over:
        print(f"          {w} words: {t}")
    bad += len(over)

    markers = [(f, m.group(0)[:76])
               for f in sorted(PAPER.glob("*.tex")) if not f.name.endswith(".unused")
               for m in re.finditer(r"%%\s*(?:TODO|FIXME|PENDING).*",
                                    f.read_text(encoding="utf-8"))]
    print(f"\nopen editorial markers: {len(markers)}")
    for f, t in markers:
        print(f"  {f.name}: {t}")

    bib = (PAPER / "refs.bib").read_text(encoding="utf-8")
    entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S)
    nopages = [k.strip() for k, b in entries if "journal" in b and "pages" not in b]
    print(f"\njournal references without page numbers: {len(nopages)} of {len(entries)}")
    if nopages:
        print("  " + ", ".join(nopages))

    print(f"\n{'STYLE CLEAN' if not bad else f'{bad} STYLE VIOLATIONS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
