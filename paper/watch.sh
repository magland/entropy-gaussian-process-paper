#!/usr/bin/env bash
# Watch paper/*.tex and rebuild paper.pdf on change.
# Polling watcher (1s) — no inotify/latexmk/entr required.
cd "$(dirname "$0")"

build() {
    local pass=1 auxsum_before auxsum_after
    auxsum_before=$(md5sum paper.aux 2>/dev/null)
    if ! pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] BUILD FAILED:"
        grep -A2 '^!' paper.log | head -20
        return 1
    fi
    # rerun while the aux file keeps changing (labels/refs settling)
    auxsum_after=$(md5sum paper.aux 2>/dev/null)
    while [ "$auxsum_after" != "$auxsum_before" ] && [ $pass -lt 4 ]; do
        auxsum_before=$auxsum_after
        pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null 2>&1
        auxsum_after=$(md5sum paper.aux 2>/dev/null)
        pass=$((pass + 1))
    done
    echo "[$(date +%H:%M:%S)] built paper.pdf ($pass pass(es))"
}

echo "Watching $(pwd)/*.tex — Ctrl-C to stop."
build
last=$(stat -c '%Y %n' ./*.tex | md5sum)
while sleep 1; do
    now=$(stat -c '%Y %n' ./*.tex | md5sum)
    if [ "$now" != "$last" ]; then
        last=$now
        build
    fi
done
