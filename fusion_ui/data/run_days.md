# Run days

One section per C-Mod run day, keyed by the first seven digits of the shot
number. Everything here is **hand-written** — there is no scraper, and the app
never goes to the network. The source is the run page on the C-Mod internal
web, e.g. for 1160616:

    https://www-internal.psfc.mit.edu/research/alcator/program/cmod_runs.php?run=1160616

which lists the miniproposals scheduled that day (each with the shot range it
owns) and the session leader's plan. **The miniproposal number and title are
quoted verbatim from that page; the paragraph under it is a summary written
here** and can be wrong in the way any paraphrase can. When a day was shared
between several miniproposals, the one named is the one whose shot range covers
the shots in the discharge database — the others are mentioned only where they
took the shots around ours.

Format, which `fusion_ui/core/rundays.py` parses and nothing else depends on:
a `## <seven digits>` heading, then one bold `**MP<number> — <verbatim title>**`
line, then free markdown until the next heading. Add a day by copying a
section; a day with no section here is still listed in the app, just without a
purpose.

## 1090813

**MP561 — GPI measurements of SOL turbulence and comparison to the SOL width**

Second run day of MP561 (S. Zweben; session leader J. Terry), all in ohmic
plasmas. Both Phantom cameras imaged the gas puff while the ASP scanned on
selected shots and the FSP dwelled at ρ ≈ 1 cm, so that the turbulence GPI sees
can be compared against the SOL width measured by the probes, the IR TV and the
edge Thomson.

## 1090826

**MP560 — High density fueling and density limits**

An afternoon half-day (M. Greenwald, shots 11–30) taken over early when the
morning's impurity-seeding miniproposal was defeated by disruptions. Helium
discharges were fuelled with a continuous density ramp up to the density limit
— reaching 90–100 % of it on shots 13–19 — while the temperature profile
evolution and the edge fluctuations were followed on the way up; ICRF was added
from shot 22 to see how close to the limit full power could still be coupled.

## 1091216

**MP571 — Further Investigation of Turbulence and Transport in Low Density Ohmic Plasmas to Resolve the Myster**

Second day of ohmic transport and turbulence studies in the low-density,
neo-Alcator regime (M. Porkolab), stepping the line-averaged density from
0.2 to 0.6 × 10²⁰ m⁻² at 3.9 T / 0.6 MA and then again at 2.6 T / 0.4 MA. PCI,
Thomson, neutrons and HIREX were the key diagnostics; GPI ran as a piggyback
from 1.4 s. (The title is truncated on the run page — it ends mid-word there.)

## 1100803

**MP570 — Boundary layer heat transport experiments in L-mode plasmas**

Ohmic L-mode density scans in three blocks (B. Labombard): 4 T / 0.8 MA,
then a few 5.4 T / 1.1 MA discharges, then the start of an 8 T / 0.8 MA scan.
Two shots at each density with ASP, FSP and WASP all scanning to resolve the
shear layer, D2 puffed for GPI at 1.2 s and the IR camera on throughout.

## 1110201

**MP644 — Edge profiles and fluctuations in EDA and ELM-free H-Modes**

A full day on the edge of H-mode plasmas (J. Hughes), contrasting EDA — where
the quasi-coherent mode regulates the pedestal — with ELM-free H-modes. The
D-port ICRF antenna was configured for frequency modulation and switched on
during selected parts of the discharge, to look for a fast RF power modulation
effect on the fluctuations and on the QCM in particular.

## 1120217

**MP667 — Commissioning of new SOL probes**

Commissioning of the A-port ion sensitive probe (D. Brunner), collecting a base
SOL data set of plasma potential and ion temperature. Starting from a
5.4 T / 0.8 MA ohmic plasma the density was scanned up to 1.6 × 10²⁰ m⁻² with at
least one good scan at each step, the point being to see T_i fall towards T_e as
collisionality rises. GPI ran alongside on the F-port probe data collected for
O. E. Garcia.

## 1120712

**MP697 — First operation of Shoelace Antenna: Survey of boundary plasma response**

Second day of the first Shoelace antenna campaign (T. Golfinopoulos), in ohmic
EDA H-mode. The antenna was driven to look for interaction with the
quasi-coherent mode, and to test whether it could drive a QCM in a discharge
where the mode had been suppressed; GPI, PCI, reflectometry and the full Mirnov
set watched the response.

## 1120814

**MP719 — Shoelace antenna operation in EDA H-mode plasmas.**

The continuation of the Shoelace campaign into a full run day of ohmic EDA
H-modes (T. Golfinopoulos, shots 1–32), reloading the 1120712 discharges. Same
question as before — how the antenna couples to the quasi-coherent mode — with
helium in the NINJA for GPI and the scanning probes on the boundary.

## 1121002

**MP723 — Inner wall erosion measured with AGNOSTIC**

The last third of a three-miniproposal day (D. Whyte, shots 25–33). Inner-wall
limited discharges with 2 MW of ICRF were run so that recycling and heating land
on the inner midplane, and AGNOSTIC — a 900 keV deuteron beam with shielded
scintillator detectors — measured the boron and oxygen surface fluence on the
centerpost between shots. Shots 25–28 fought vertical disruptions; 29 and 31–33
were the good limited discharges.

## 1140228

**MP725 — Development of inner-wall limited discharges for SOL heat and particle transport studies**

Development of inner-wall limited (IWL) discharges (B. Labombard; session leader
J. Terry) at 5.4 T / 0.8 MA, κ ≈ 1.2, raised later to 6.4 T / 1.0 MA. The outer
gap was opened to ~1.8 cm to put the LCFS in the GPI field of view, and GPI,
gas-puff CXRS and all three scanning probes recorded the profiles. The immediate
question was whether C-Mod sees the "narrow feature" in the inner-wall heat flux
profile that other machines reported, which ITER's inner-wall tile shaping
depends on.

## 1140612

**MP741 — SOL heat and particle transport studies of inner-wall limited discharges**

The afternoon half of a split day (J. Terry, shots 19–32; the morning was
runaway-electron mitigation under MP732). It picks up the inner-wall limited
programme begun on 1140228: the inboard SOL measured with the inboard scanning
probe and inboard GPI, the outboard SOL with the outer probes and the outboard
GPI, on the same discharge.

## 1140613

**MP734 — GAM and zonal flow energy transfer and their relation to confinement**

Day one of two (I. Cziegler). GPI was used to establish the edge flow structure
in I-mode and in ICRF L-mode, measuring the nonlinear kinetic energy transfer
between the broadband turbulence and the GAM (I-mode) or the zonal flow
(L-mode), and following that transfer through the L–I and I–H transitions with
time resolution.

## 1140619

**MP741 — SOL heat and particle transport studies of inner-wall limited discharges**

A full run day continuing the half-day of 1140612 (J. Terry), starting at 4 T.
Inner-wall limited discharges again, with helium puffed for GPI and the scanning
probes plunging repeatedly, to characterise the inboard and outboard SOL of the
same plasma at once.

## 1140827

**MP761 — Density limit experiments in support of simulation validation**

The afternoon half-day (I. Cziegler, shots 17–30; the morning was lower-hybrid
parametric instabilities under MP760). A density scan at 4.0 T / 0.6 MA with the
outer gap held near 1.2 cm, approaching the density limit with outboard GPI,
core and edge CXRS, Thomson and the mirror Langmuir probe as the critical
diagnostics — data taken specifically to validate turbulence simulations.

## 1150618

**MP761 — Density limit experiments in support of simulation validation**

The afternoon of a split day (session leader J. Terry, shots 20–37); the morning
was Shoelace antenna discharge development under MP772. The density-limit scan
of 1140827 was continued in the ohmic EDA H-mode / L-mode targets developed that
morning, at 0.75 MA with helium puffed from the C-port NINJA for GPI. This is
the day the group's one ASP probe file comes from.

## 1150916

**MP734 — GAM and zonal flow energy transfer and their relation to confinement**

The low-field segment of the GAM miniproposal (I. Cziegler), at 2.8 T / 0.6 MA
with ICRF in D(H) second harmonic. Earlier days had shown an edge coherent mode
that can exist in L-mode before the GAM or the I-mode appears; running at
reduced field separates that mode from the GAM, and lowers the H-mode threshold
so that I–H and direct L–H transitions are easier to reach.

## 1160616

**MP800 — Scrape-off layer fluctuation statistic in ohmic L- and EDA H-modes**

The group's own miniproposal (O. E. Garcia; session leader B. Labombard):
characterise the statistical properties of cross-field transport in the SOL of
ohmic L-mode and EDA H-mode plasmas using GPI together with the scanning and
divertor mirror Langmuir probes. This day was an ohmic L-mode density scan to
high Greenwald fraction at 0.55 MA / 5.4 T, in seven NL04 steps from n/n_G ≈ 0.12
upwards, with the ASP dwelling just inside the limiter radius and helium puffed
for GPI only while the MLP was scanning.

## 1160629

**MP751 — Investigating the physics of the heat flux channel width in L and H-modes**

High-resolution profiles across the last closed flux surface with the scanning
MLP (B. Labombard), tracking the heat flux width in ohmic discharges at 0.55 MA
and 1.1 MA. Divertor dissipation was set by feedback-controlled N2 seeding off
the surface thermocouples, to see how it feeds back on the upstream heat flux
width.

## 1160706

**MP681 — Evaluate field-aligned antenna performance**

A run day on the field-aligned ICRF antenna (S. Wukitch), evaluating its
electrical performance and its interaction with the boundary plasma. Helium was
loaded in the C-port NINJA for GPI, which rode along on the antenna discharges.

## 1160926

**MP815 — Characterization of ICRF antenna: electrical performance, impurity contamination and SOL interaction**

The afternoon block (S. Wukitch, shots 17–39), following on from 1160706. The
strike point was raised to optimise the GPI view, then the plasma response was
scanned against the central/outer ICRF power ratio, and N2 was puffed at low
power to measure impurity penetration.

## 1160927

**MP828 — Documenting the effect of divertor geometry on upstream scrape-off layer profiles**

Density-stepped shots with the strike point first on the vertical target plate
and then up on the shelf (A. Kuang), to separate what divertor geometry does to
the upstream profiles: the plasma potential profile and its rollover voltage,
and the dependence of SOL transport on divertor collisionality in vertical
versus horizontal plate geometry. ASP with the MLP, the divertor probe arrays,
edge Thomson, GPI and the x-point camera all ran.

## 1160929

**MP750 — Investigation of the mode structure of the WCM with a scanning Mirror Langmuir probe**

The first half of the day (B. Labombard, shots 1–17). The MLP had already
resolved the mode structure of a 110 kHz quasi-coherent mode — density, electron
temperature and plasma potential sampled at 1.1 MHz inside the mode layer, with
the electric field measured simultaneously so the phase velocity could be
converted to the plasma frame. The goal here was the same measurement for the
weakly coherent mode, with the GPI fast-diode system and reflectometry alongside.
