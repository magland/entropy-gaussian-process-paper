#!/usr/bin/env bash
# Build a PDF from one of the .tex files in this directory.
#   ./build.sh                     -> note_for_colleagues.pdf
#   ./build.sh paper               -> paper.pdf
# Reruns pdflatex until the .aux file settles, so \ref and \eqref resolve.
set -u
cd "$(dirname "$0")"

name="${1:-note_for_colleagues}"
name="${name%.tex}"

if [ ! -f "$name.tex" ]; then
    echo "no such file: $name.tex" >&2
    exit 1
fi

run() {
    pdflatex -interaction=nonstopmode -halt-on-error "$name.tex" >/dev/null 2>&1
}

pass=1
before=$(md5sum "$name.aux" 2>/dev/null)
if ! run; then
    echo "BUILD FAILED:" >&2
    grep -A2 '^!' "$name.log" | head -20 >&2
    exit 1
fi
after=$(md5sum "$name.aux" 2>/dev/null)
while [ "$after" != "$before" ] && [ $pass -lt 4 ]; do
    before=$after
    run
    after=$(md5sum "$name.aux" 2>/dev/null)
    pass=$((pass + 1))
done

grep -q 'LaTeX Warning: Reference' "$name.log" && echo "warning: unresolved references remain"
grep -c Overfull "$name.log" | grep -qv '^0$' && grep -n Overfull "$name.log"

echo "built $name.pdf ($pass pass(es))"
