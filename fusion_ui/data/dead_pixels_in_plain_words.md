**In plain words.** A pixel is *live* when it sees the plasma, and *dead* when its channel carries only electronic
noise, or nothing. Preprocessing fills a dead pixel in from its neighbours. Two tests decide, both on how the signal
fluctuates in time and not on how bright it is:

1. **It fluctuates like plasma.** Electronic noise is equally strong at all frequencies. Plasma fluctuations are far
   stronger at low frequency (1–20 kHz) than at 300–900 kHz, where only digitizer noise is left. A pixel with far more
   low-frequency than high-frequency power is live.
2. **It follows a live neighbour.** A dim pixel can fail the first test and still be live: its slow signal moves with
   a neighbouring pixel already found live, and carries a real fraction of that pixel's amplitude. This repeats until
   no pixel joins.

Every other pixel is dead.

**Run-day rule.** The hardware does not change within a run day, so a pixel found dead in at least a third of the
day's shots is dead in all of them. Run day 1160616 uses the hand-made mask instead, which these two tests reproduce.
