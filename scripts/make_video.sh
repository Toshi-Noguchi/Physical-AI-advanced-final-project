#!/usr/bin/env bash
# Recorded frames (run_cell.py --record, test_vla_pick.py --video) -> H.264 MP4 at 30 fps. With SPEED other than 1
# the video plays that much faster and says so in the top-right corner ("2x speed"), as the course asks.
#
#   scripts/make_video.sh outputs/cell/frames outputs/cell/cell.mp4
#   SPEED=2 scripts/make_video.sh outputs/cell/frames outputs/cell/cell_2x.mp4
set -euo pipefail

FRAMES=${1:?usage: make_video.sh FRAMES_DIR OUT.mp4}
OUT=${2:?usage: make_video.sh FRAMES_DIR OUT.mp4}
SPEED=${SPEED:-1}
FPS=${FPS:-30}  # the rate the frames were recorded at
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FONT=$REPO/assets/fonts/DejaVuSans-Bold.ttf

vf="scale=trunc(iw/2)*2:trunc(ih/2)*2"  # H.264 with yuv420p needs even sizes
if [[ "$SPEED" != 1 ]]; then
  vf+=",drawtext=fontfile=$FONT:text='${SPEED}x speed':x=w-tw-28:y=24:fontsize=36:fontcolor=white"
  vf+=":box=1:boxcolor=black@0.55:boxborderw=10"
fi
mkdir -p "$(dirname "$OUT")"
ffmpeg -hide_banner -loglevel error -y \
  -framerate "$(awk "BEGIN { print $FPS * $SPEED }")" -pattern_type glob -i "$FRAMES/*.jpg" \
  -vf "$vf" -r "$FPS" -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart "$OUT"
seconds=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT")
echo "$OUT: $(find "$FRAMES" -maxdepth 1 -name '*.jpg' | wc -l) frames, ${seconds%.*} s"
