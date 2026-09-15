# Achieving the Quantum Limits with Twin-Field Sensors

This repository contains the Julia source code for the two-level-system examples presented in our work ``Achieving the Quantum Limits with Twin-Field Sensors".

http://arxiv.org/abs/2609.07517

## Model

We consider a pair of driven two-level systems with opposite detunings,

$$
H_1 = -\delta \sigma_1^+\sigma_1^- + \frac{\Omega}{2} (\sigma_1^+ + \sigma_1^-),
$$

$$
H_2 = +\delta \sigma_2^+\sigma_2^- + \frac{\Omega}{2} (\sigma_2^+ + \sigma_2^-),
$$

with independent spontaneous-emission channels

$$
c_1=\sqrt{\gamma}\sigma_1^-,
\qquad
c_2=\sqrt{\gamma}\sigma_2^-.
$$

The opposite detunings constitute the anti-frequency twin-field configuration considered in the paper.

The basis of the joint two-level system is ordered as

$$
|ee\rangle,\quad |eg\rangle,\quad |ge\rangle,\quad |gg\rangle.
$$

## Source Files

The repository contains four main Julia scripts:

### `two_level_qfi_t.jl`

Calculates the quantum Fisher information (QFI) as a function of evolution time using the generalized-density-matrix method.

The script evaluates both:

* the total QFI of the joint system-output state;
* the QFI contained in the emitted field.

The field QFI is obtained from the trace norm of the generalized density matrix.

### `two_level_qfi_rate.jl`

Calculates the asymptotic QFI growth rate from the dominant eigenvalue of the generalized Liouvillian.

This script is useful for studying the long-time information rate without explicitly propagating the dynamics to long evolution times.

### `two_level_cfi_t_pc.jl`

Calculates the classical Fisher information (CFI) obtained from continuous photon-counting records.

The script compares local detection of the two emission channels with collective detection after interference of the output fields.

Quantum trajectories are generated using the non-Hermitian no-jump evolution and stochastic quantum jumps.

### `two_level_cfi_t_hd.jl`

Calculates the CFI obtained from continuous homodyne detection.

The script considers local and collective homodyne measurements with different local-oscillator phases. The CFI is evaluated from the trajectory score associated with the continuous measurement record.

## Other Models

The repository provides the two-level-system implementation as the representative example of the numerical methods used in the paper.

The other models discussed in the paper are evaluated using the same QFI and CFI procedures. Their numerical implementations differ primarily in the system Hamiltonian, Hilbert space, and corresponding emission operators. These model-specific modifications do not change the underlying numerical methods demonstrated by the four scripts provided here.

## Requirements

The codes are written in Julia.

The main packages used by the scripts include:

* `LinearAlgebra`
* `SparseArrays`
* `Distributed`
* `Plots` or `CairoMakie`
* `LaTeXStrings`
* `DataFrames`
* `CSV`
* `Arpack`

The exact packages required depend on the individual script.

## Running the Codes

Each script can be executed directly from Julia, for example:

```bash
julia two_level_qfi_t.jl
```

and

```bash
julia two_level_cfi_t_pc.jl
```

The scripts support parallel computation through Julia's `Distributed` standard library. Worker processes are initialized within the scripts, so it is not necessary to start Julia with the `-p` option.

The main physical and numerical parameters can be modified in the script entry point, including the detuning, Rabi frequency, spontaneous-emission rate, finite parameter shift, total evolution time, number of trajectories, and number of sampled time points.

## Output

Depending on the script, numerical results are saved as CSV files and the corresponding figures are saved as image files in the specified output directory.

For stochastic CFI calculations, the numerical accuracy improves with the number of quantum trajectories. For finite-difference QFI calculations, convergence with respect to the parameter displacement should be checked.

## Reproducibility

The two-level-system codes provided here contain the complete numerical procedures used for the corresponding QFI and CFI calculations in the paper.

For the other physical models, the same numerical procedures are used with the Hamiltonian and system operators replaced by those of the corresponding model.

## Citation

If you use this code, please cite the paper

## License

Please see the `LICENSE` file for the terms of use.
