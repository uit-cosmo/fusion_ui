
These are some of the issues I encountered while playing with the UI.

Certain statistic plots (pdf, psd, ccf) that take as input a time series (a trace), should be able to run on all current diagnostics, for any time series (any pixel in the case of apd/phantom, or any probe in the case of the others), and for any time window. Additionally it would be convenient to plot serveral at the same time, labelled maybe with the magnetic coordinates. 

I am a bit unsure how this should look at the end, so please come with ideas. A possibility is that we have a statistics plot type that, say pdf that, given a shot, allows you to throw in time series that you can select from all the diagnostics (maybe a UI with a pixel selection would work nice here for the case of APD/phantom). Additionally, it should have a time window that centers the traces.

