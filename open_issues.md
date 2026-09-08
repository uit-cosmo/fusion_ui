
These are some of the issues I encountered while playing with the UI.

~~Certain statistic plots (pdf, psd, ccf) that take as input a time series (a trace), should be able to run on all current diagnostics, for any time series (any pixel in the case of apd/phantom, or any probe in the case of the others), and for any time window. Additionally it would be convenient to plot serveral at the same time, labelled maybe with the magnetic coordinates.~~

**Done — see the Statistics page** (`fusion_ui/pages/4_statistics.py`, plan in
`issue_implementation.md`): a basket of traces from any shot and any diagnostic
draws together under PDF, PSD, ACF or CCF over an absolute time window,
labelled with magnetic coordinates (`R−R_sep` for pixels, window-mean ρ for
probe channels).

I am a bit unsure how this should look at the end, so please come with ideas. A possibility is that we have a statistics plot type that, say pdf that, given a shot, allows you to throw in time series that you can select from all the diagnostics (maybe a UI with a pixel selection would work nice here for the case of APD/phantom). Additionally, it should have a time window that centers the traces.

Follow-ups:

- A **moments** statistic — mean, std, skewness, kurtosis per trace as a table
  — is about thirty lines on top of this machinery and is the obvious fifth
  `StatSpec`.
- The basket is the natural place to eventually carry a trace into a cached
  analysis: a "compute `taud_psd` for every trace in this basket" button would
  join the two halves of the app.
