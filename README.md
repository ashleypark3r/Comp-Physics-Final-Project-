# Comp-Physics-Final-Project-
Saturn's Ring Dynamics 

**Goal:** Compare the phase of the dominant NEV modulation across BetCen occultations and density waves. The method is an **unweighted first-order sinusoid** fitted to 100-photon-sample-bin Sheppard-corrected NEV. This estimates the phase of the **first harmonic**, which may differ from the highest individual point or the true maximum if the profile is not sinusoidal.

**Instructions:** Change `occstring`, `wavestring`, file directories, and the **four crest guesses** for each analysis. Run from top to bottom. Check the crest plot before accepting any result. The notebook reports approximate bootstrap confidence intervals, but these depend on the sinusoidal model and resampling assumptions; leave-one-cycle-out tests are separate sensitivity checks.
