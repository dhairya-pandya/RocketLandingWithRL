import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # render tests never open a window
