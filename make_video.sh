ffmpeg -loop 1 -i bild.png -i musik.mp3 \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" \
  -c:v libx264 \
  -preset medium \
  -crf 18 \
  -tune stillimage \
  -c:a aac \
  -b:a 192k \
  -pix_fmt yuv420p \
  -r 30 \
  -shortest \
  -movflags +faststart \
  fuer_elise_group.mp4
