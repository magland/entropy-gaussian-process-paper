#!/usr/bin/env bash
# Assemble the arXiv submission: the sources the paper needs and nothing else,
# checked by a clean standalone build before packing.
#   ./make_arxiv.sh   -> arxiv-submission.tar.gz
set -eu
cd "$(dirname "$0")"

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/figures"
cp paper.tex compressors_table.tex family_curves_tables.tex "$stage/"
cp figures/*.png "$stage/figures/"

(
    cd "$stage"
    for _ in 1 2 3; do
        pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null 2>&1 \
            || { echo "standalone build failed:" >&2; grep -A2 '^!' paper.log >&2; exit 1; }
    done
    if grep -q 'LaTeX Warning: Reference' paper.log; then
        echo "unresolved references in standalone build" >&2; exit 1
    fi
    rm -f paper.aux paper.log paper.out paper.pdf
)

tar czf arxiv-submission.tar.gz -C "$stage" .
echo "wrote arxiv-submission.tar.gz ($(du -h arxiv-submission.tar.gz | cut -f1))"
