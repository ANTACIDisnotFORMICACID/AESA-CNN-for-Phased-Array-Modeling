# V2.6 known limitations

1. The fixed 47 GHz array has a 6.8 mm spacing (about 1.07 wavelengths) and a
   broad/ambiguous full-sphere response.  In the completed production-setting
   three-direction oracle spot check, continuous amplitude/phase margins were
   `0.0758 dB`, `-0.6752 dB`, and `-1.9752 dB`, all below the required `3 dB`.
   V2.6 therefore reports best effort and does not claim accurate 2-D scanning.
2. The complete 64x64 continuous and INT8 oracle map has not yet been computed.
   The implementation is resumable, but release teacher generation, full
   96/24/48-state training, checkpoint selection, and test acceptance must wait
   for that map.  A spot check must not be extrapolated into a formal whole-domain
   feasibility proof.
3. The current host has no discoverable Docker CLI, `pytorch_nndct`, or
   `vai_c_xir`.  Static graph audit passes, but PTQ, XModel export, and KV260
   partitioning remain blocked.  Nothing was installed or downloaded.
4. HFSS active-element patterns are a solved simulation backbone.  They do not
   include the final RF feed board, connector repeatability, chamber fixtures,
   thermal drift, measurement synchronization, or FPGA/OTA latency.
5. Calibration measurements are complex IQ measurements.  A power-only receiver
   cannot use this estimator without a separate phase-retrieval design.
6. The 3 dB criterion is evaluated outside an 8-degree exclusion cone.  For a
   four-element aperture the beam itself can remain broad at that boundary; a
   failed margin therefore includes finite-aperture beamwidth as well as grating
   lobes.  The release threshold is not silently relaxed.
7. V2.5 remained read-only.  Its 35 unit tests and V2.4 frozen bundle pass, but
   the V2.5 full-pattern byte-exact JSON check differs in one FP64 value by
   `4.93e-32`; this is recorded rather than overwriting the frozen artifact.

