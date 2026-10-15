# V2.12 implementation verification — 2026-09-10

## Completed checks

- Standalone acceptance suite: 18/18 passed.
- Every tool entry point imports and returns `--help` successfully.
- V2.10 and V2.11 were not edited.
- Actual HFSS preparation produced 32,400 unique strict-front-hemisphere directions,
  with 22,600/4,900/4,900 train/validation/test directions.
- Training dataset contains 96-command scalar amplitudes and explicitly declares
  `phase_available=false`; complex `E_phi` is stored only under the audit path.
- A four-direction reconstruction spot-check agreed with `abs(g^T w)` to a maximum
  absolute difference of `1.91e-6` after float32 serialization.
- The command split is 64/16/16 and the train pair graph is connected.
- Real-data smoke checks confirmed exact codebook round-trip, E2-F4 zero-gate equality
  to E0, E2-F4/E0-M4 parameter matching, G2 zero-gate equality to G1 and Hermitian Q.

## Filtration basis audit

| Degree | Raw channels | Retained ranks | Raw Gram condition | Retained condition | Transformed condition |
|---:|---:|---|---:|---:|---:|
| 4 | 25 | 1,3,5,7,9 | 266,464.96 | 286.44 | 1.000000000013 |
| 5 | 36 | 1,3,5,7,9,11 | 9,158,147.18 | 1,205.84 | 1.000000000154 |

Both L4 and L5 retained all raw modes under the frozen squared-singular-value
threshold. The transformed design is numerically orthonormal under the empirical
equal-loss training measure.

## Gauge-invariant HFSS audit

For raw-SSH L12 fitted to `Q_tilde=D_r Q D_r^H=a^*a^T` with physical solid-angle
weights:

- cumulative energy through L4: 0.93144283;
- cumulative energy through L5: 0.98023017;
- cumulative energy through L10: 0.99999986;
- normalized gradient energy: 9.24266326.

These figures describe the gauge-invariant operator, not the non-identifiable latent
complex envelope spectrum, and are not used as a training loss.

## Diagnostic smoke runs

- Eight-direction rank audit, validation commands only: rank-1 accepted; rank-2
  median improvement 0 dB; test commands remained sealed. This is an execution
  smoke test, not the full contract decision.
- Two-step, one-restart O1 smoke: the pipeline completed without opening test
  directions. It did not pass the recommendation gate and must not be interpreted
  as the frozen 5 x 2,000-step O1 result.

## Deliberately not executed

- The eight Stage-1 recipes times ten paired seeds.
- Full-direction rank-spectrum inversion and G3 target generation.
- Ten-seed G1/G2/G3 training and G3 lambda selection.
- Five-restart, 2,000-step O1 optimization and test Monte Carlo.
- Any external-test model evaluation.

Those are long-running experiments. Their frozen entry points and sealed-test guards
are implemented, but no result is claimed before the configured campaigns finish.
