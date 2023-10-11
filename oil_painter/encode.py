
import subprocess as sp
import os
import sys




for d in os.listdir("frames"):
    if d==".DS_Store":
        continue
    cmd = f"ffmpeg -framerate 30 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p {d}.mp4"

    d = f"./frames/{d}/"

    if (os.path.isfile(f"{d}video.mp4")):
        print("SKIPPING VID:", d)
        continue

    print("NOW ENCODING: ", d)
    sp.check_call([cmd], cwd=d, shell=True, stderr=sys.stderr, stdout=sys.stderr)



