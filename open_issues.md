
These are some of the issues I encountered while playing with the UI.

# Issues

1. In the main page, present an overview of the stored data. It should be easily fetched from the cmod archives, but ideally it should also contain an overview of academic articles that use that data.
2. Single Shot: APD doesn't change the pixel when clicking on the frame
3. In all plots that need to coarsen the data to plot due to too high number of samples, resample to a higher resolution when zooming in.
4. In the Tracked trajectories, upper left plot, the reference pixel is way out of the field of view. The reason is that its position is given in meters but plotted as if it was centimiters. Same applies to the R* and Z* in the plots on the right.

# TODOS for later

1. Some diagnostics should be able to be run on several pixels at the same time. Have a pixel selection window, for example a plot of all the pixel locations where the pixels can be selected with a rectangle. Once they are selected, a plot type is choosen (for example, PSD fit), and then the method is applied to all selected pixels and presented in a single plot
