# A minimizing movement framework for anisotropic curve shortening flows

This directory contains the Firedrake code accompanying the paper
“A minimizing movement framework for parametric finite element methods about
anisotropic curve shortening flows” by Wei Jiang, Chunmei Su, and Ganghui
Zhang.

The code implements the semi-implicit L2 minimizing-movement parametric finite
element method (MMS-PFEM) for anisotropic curve shortening flow (CSF) and its
area-preserving counterpart (AP-CSF). Weak anisotropies are treated directly
by the modified mixed energy. Strong anisotropies are treated by the convex
splitting

    gamma = gamma_c - gamma_e,
    gamma_c = gamma + C |.|,
    gamma_e = C |.|.

## Scripts

| script | experiment |
| --- | --- |
| evolution_csf.py | CSF evolution, energy decay, mesh ratio, Newton iterations |
| evolution_apcsf.py | AP-CSF evolution, energy decay, area conservation, mesh ratio |
| eoc_csf.py | coupled space-time convergence for CSF with tau=h^2 |
| eoc_apcsf.py | coupled space-time convergence for AP-CSF with tau=h^2 |
| figure_csf_evolution.py | redraw the two panels of fig:csf_evolution |
| wulff_shapes.py | generate Cahn-Hoffman/Wulff boundary data |
| wulff_comparison.py | compare long-time AP-CSF profiles with Wulff shapes |
| strong_anisotropy_cusps.py | C-dependence and mesh-refinement cusp comparison |
| strong_apcsf_wulff.py | strongly anisotropic AP-CSF with Wulff overlay |

Five helper modules are used by the experiment scripts:

| module | purpose |
| --- | --- |
| mms.py | shared Firedrake CSF and AP-CSF variational solvers |
| io_tools.py | portable compressed-NPZ result storage |
| plotting.py | evolution and diagnostic figures |
| convergence.py | manifold distance, aligned time steps, EOC tables |
| wulff.py | NumPy implementation of gamma_hat and Cahn-Hoffman curves |

## Dependencies

- Firedrake, including PETSc and petsc4py
- NumPy
- Matplotlib
- Shapely, recommended for robust polygon symmetric differences
- SciPy, required by strong_apcsf_wulff.py

Firedrake is not a normal PyPI dependency. Install it using the official
Firedrake instructions and activate that environment before running the code.
For example:

    source /path/to/venv-firedrake/bin/activate

## Running

Run the scripts as plain Python files from this directory so their local helper
imports resolve:

    python evolution_csf.py
    python evolution_apcsf.py
    python eoc_csf.py

The scripts intentionally have no command-line interface. Edit the clearly
marked Parameters block near the top of a script to change cases, mesh sizes,
time steps, final times, or output choices.

All generated files are written under output/.

## Reproducing the paper

### CSF and AP-CSF convergence tests

The default case lists in the two EOC scripts contain the four smooth
anisotropies used in the paper:

1. isotropic;
2. q-fold with (q,beta)=(4,0.03);
3. Riemannian with G=diag(1,2);
4. l4-norm.

Run:

    python eoc_csf.py
    python eoc_apcsf.py

Each script uses N=16,32,64,128,256,512 and tau=h^2. It writes one error table
and one observed-order table per case.

### Long-time Wulff comparisons

The numerical AP-CSF curves at T=2 are included under
precomputed/wulff_comparison, so this command only redraws the four panels:

    python wulff_comparison.py

To generate the corresponding Wulff boundary tables:

    python wulff_shapes.py

### Strongly anisotropic cusp comparison

The curves for N=128,256,512 and C=0.2,1,10 are included. Run:

    python strong_anisotropy_cusps.py

Set RUN_SOLVER_IF_MISSING=True in its Parameters block to recompute missing
data rather than reading the included NPZ files.

### CSF structure-preserving evolution figure

The four complete histories used by fig:csf_evolution are included under
precomputed/csf_evolution. Redraw both image files with:

    python figure_csf_evolution.py

The figure legend uses the current manuscript notation:

    q-fold, (q,beta)=(4,0.03)
    q-fold, (q,beta)=(4,0.1), C=0.5

To recompute the four histories, run:

    python evolution_csf.py

Then point DATA_FILES in figure_csf_evolution.py at the four generated
output/csf_<case>/result.npz files.

### Strongly anisotropic flower and AP-CSF experiments

For the flower experiment, set:

    CASES_TO_RUN = ["qfold_q2_strong_flower"]

in evolution_csf.py.

For the area-preserving ellipse experiment, run evolution_apcsf.py with:

    CASES_TO_RUN = ["qfold_q2_strong_ellipse"]

The precomputed history can be plotted with the theoretical Wulff curve using:

    python strong_apcsf_wulff.py

## Per-script reference

### evolution_csf.py

The Parameters block contains:

    REUSE_DATA    load an existing result.npz when available
    CASES_TO_RUN  list of named configurations
    N             number of periodic interval elements
    dt            time-step size
    T_final       final time
    snapshot_times
    C_override    optional convex-splitting constant

The four default cases reproduce fig:csf_evolution. The additional isotropic
and strong q=2 flower configurations are retained for the other numerical
experiments.

Output layout:

    output/csf_<case>/result.npz
    output/csf_<case>/evolution.png
    output/csf_<case>/diagnostics.png

### evolution_apcsf.py

The parameter layout is the same as evolution_csf.py. The default run includes
the isotropic ellipse and the strongly anisotropic q=2 ellipse. The mixed
Firedrake space consists of the curve coordinates and one real-valued
Lagrange multiplier.

Output layout:

    output/apcsf_<case>/result.npz
    output/apcsf_<case>/evolution.png
    output/apcsf_<case>/diagnostics.png

### eoc_csf.py and eoc_apcsf.py

The Parameters blocks specify:

    REFINEMENTS = [16, 32, 64, 128, 256, 512]
    DT_FACTOR = 1
    CASES_TO_RUN
    test_times

For each adjacent pair N and 2N, the error is the area of the symmetric
difference between the two polygonal curves. Shapely is used when installed;
the built-in clipping fallback is suitable for convex curves.

### figure_csf_evolution.py

This is a plotting-only script and does not import Firedrake. It reads the four
included NPZ histories and writes:

    output/csf_combined_evolution.png
    output/csf_combined_quantities.png

### wulff_comparison.py

The theoretical boundary is xi(N)=grad gamma_hat(N), evaluated by centered
finite differences and rescaled to the area of the numerical AP-CSF curve.
It writes:

    output/wulff_compare_iso.pdf
    output/wulff_compare_kfold.pdf
    output/wulff_compare_riemannian.pdf
    output/wulff_compare_l4.pdf

The legacy word kfold is retained only in this output filename because the
current LaTeX draft includes that filename. Figure labels and mathematical
parameters use q-fold notation.

### strong_anisotropy_cusps.py

The default Parameters block uses:

    (q,beta) = (2,0.4)
    C = 0.2, 1, 10
    N = 128, 256, 512
    T = 0.03

It writes the three PDF files included by the manuscript.

## Output and data format

All newly computed histories are stored as compressed NPZ archives. They
contain numeric arrays and JSON metadata and are loaded with
allow_pickle=False. This replaces the trusted-local pickle caches used during
development.

The precomputed directory contains only data required to redraw expensive
paper figures. Generated plots, solver output, and new simulation histories
belong under output/, which is excluded by .gitignore.

## Notes on computational cost

figure_csf_evolution.py, wulff_comparison.py,
strong_anisotropy_cusps.py, and strong_apcsf_wulff.py use included numerical
data and finish quickly.

The evolution scripts may take minutes. The full EOC scripts loop over six
refinements and four anisotropies and can take hours, especially for AP-CSF.
For a quick smoke test, shorten CASES_TO_RUN and use:

    REFINEMENTS = [16, 32]

or reduce N and T_final in the relevant Parameters block.
