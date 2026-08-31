# Experimental bSWAP calibration building blocks

`qubex.contrib.experiment.bswap_calibration` contains experimental, additive
building blocks for reproducible, measurement-driven calibration. Import the
focused modules explicitly. No import opens a device connection.

The operator owns the connection, hardware settings, authorization, time budget
and notebook. These functions do not reserve hardware, reconnect, or promote
results into shared configuration. Use one live connection for phase calibration,
independent validation and benchmarking.

## Control and phase conventions

- `amplitude` is a dimensionless hardware command, not GHz.
- `frequency_ghz` is the delivered carrier. Complete duration includes two ramps.
- The SQUAD constructor converts amplitude and transition-minus-drive detuning
  to rad/ns. `K = 2*pi*r`, where `r` is the declared same-path Rabi conversion in
  GHz/command; final `scale=1/K` converts **both** I and Q to commands.
- `cd_strength` is dimensionless. Analytic unit strength is an initializer,
  not a claim that the physical hardware implements exact multilevel CD.
- `design_delta_scale` is a positive, empirical shape knob. It does not change
  the physical carrier or measure an actual detuning/gain. Keep K fixed while
  changing this knob; do not mix unit conversion with correction strength.
- Use an explicit window dictionary. Tukey positions shape a normalized ramp
  window, not two independent outer ramp durations.

The pulse convention is I+iQ with transition-minus-drive detuning and the
negative CD quadrature. `make_squad_pulse` validates the sampled complex peak
and the native time grid. A correctly sampled pulse is not evidence of the
analog transfer function or physical gate fidelity.

Full and square-root recipes are separate. Their measured phase decompositions
contain active Pre, active Post and passive Post (passive Pre is the zero gauge).
Do not substitute inverse VZ corrections for those measured angles.

`compile_campaign` maintains one logical phase per qubit, including prior gate
corrections and arbitrary VZs. Both one-qubit axes and the separate two-photon
drive use that state. The placement rate comes from declared reference/carrier
frequencies, not an empirically fitted circuit-return rate. The main and optional
siZZle tones receive the same logical phase shift, preserving their calibrated
relative phase. Although ZZ commutes with local Z, shifting only the main tone
would change the physical siZZle control.

The compiler accepts independently calibrated full/root carriers. It materializes
their offsets relative to fixed target references, including absolute-time phase.
Pass those same fixed target references to `Experiment.measure`; passing a recipe
carrier again would double-apply the offset.

## Measurement and calibration

- `chevron_fit`, `duration_fit`, and `local_fit` provide phenomenological fits,
  not hardware or microscopic-mechanism proofs. Duration fits use flat-top dwell
  with a fitted ramp offset; choose full/root angle times on the native grid.
- Hold each chosen duration fixed during its local amplitude/frequency map.
  Reopen duration when starting a different ramp/amplitude family or when
  independently observed failure requires it.
- `tomography` separates local phases from residual integrated ZZ. Preparation
  and analysis must use fresh production DRAG gates.
- `measurements` takes an already connected Experiment and explicit qubit and
  channel labels. It preserves raw IQ/counts before analysis. Its optional
  deadline never substitutes for resource authorization.
- Prepared-state response inversion uses `C[reported, prepared]`. It may include
  thermal population and preparation errors; it is not an independently known
  detector response. Keep negative and greater-than-one normalized values,
  record matrix conditioning, and retain the raw analysis as a separate result.
- `drift` measures repeated, fixed-waveform response peaks. It records timestamps,
  baseline identities and fit uncertainty; a shifted peak does not identify
  amplitude drift as its cause.

The siZZle optimizer requires a signed ZZ bracket and independent null validation
on each exact final waveform. It can fail without returning a qualified recipe.
Retune exchange and local phases with the tone on; full and square-root nulls
are not interchangeable. The null refers to pulse-integrated ZZ, not necessarily
the instantaneous coefficient at every point on the ramp.

The final SQUAD search is a bounded, measurement-driven local optimization.
Begin with a small shape-parameter set and fixed duration/ramp family. Recheck
exchange, local phases and the siZZle null after shape changes. Training data
select candidates; fresh held-out data evaluate frozen candidates. A lower
command amplitude or longer ramp is a separately recorded family, not a hidden
change to the benchmark being compared.

## Benchmark contracts

Native full bSWAP, two sequential square roots, and echoed XX90 have distinct
waveforms and gate durations. A square-root pair IRB result is a block result;
taking its square root does not measure single-root fidelity.

Standard IRB uses an explicit ideal Clifford inverse. Physical residual ZZ is
part of the error. Matrix sign/inverse checks, sampled waveforms, independent
phase-sensitive prefix checks and a valid reference decay are prerequisites.
Invalid, floor-collapsed or poorly identified fits are not fidelity results.
Seed-bootstrap intervals do not cover reference/gate-dependent systematic error.

XEB uses independent 8-axis DRAG X90 and 8-angle VZ choices on each qubit,
plus a terminal local analysis layer. A square-root circuit contains one root
per cycle. Freeze the raw model's ZZ and local phase calibration before acquiring
benchmark shots; compare that fixed raw target and the zero-ZZ target using the
same records. Do not fit target parameters on benchmark data and then report
the fit as independently measured fidelity. XEB cycle decay does not by itself
establish unconditional single-gate average fidelity or resolved leakage.

## Validation boundary

Tests include numerical fit counterexamples, native-Clifford matrix closure,
actual flattened I/Q carrier/phase replay, raw-versus-unclipped response
inversion, finite-shot phase cycling and bounded count-only optimization.
They are offline software/estimator evidence. Hardware qualification remains
a separate, saved notebook result with its own baseline and time window.
