
These are some of the issues I encountered while playing with the UI.

# Issues

1. In the main page, present an overview of the stored data. It should be easily fetched from the cmod archives, but ideally it should also contain an overview of academic articles that use that data.
2. Single Shot: APD doesn't change the pixel when clicking on the frame
3. In all plots that need to coarsen the data to plot due to too high number of samples, resample to a higher resolution when zooming in.
4. In the Tracked trajectories, upper left plot, the reference pixel is way out of the field of view. The reason is that its position is given in meters but plotted as if it was centimeters. Same applies to the R* and Z* in the plots on the right.

# TODOS for later

(none — the multi-pixel TODO below shipped as Many mode on the single-shot page)

# Done

1. Several pixels at the same time. The single-shot page offers eligible specs
   (cached, with `refx`/`refy` in their params) as a One/Many toggle: Many
   selects a rectangle of pixels on the pixel map, runs the analysis once per
   pixel through the shared cache, and draws every pixel on one axis
   (`core/multipixel.py`, optional per-spec `overlay`, worked example in
   `plots/spectra.py`). Known limitation: the multi-shot page still lists the N
   pixel-runs as N separate sources rather than aggregating the rectangle.
