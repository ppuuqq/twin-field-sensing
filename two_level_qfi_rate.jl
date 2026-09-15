"""
Asymptotic Quantum Fisher Information Rate
for a Pair of Driven Two-Level Systems

Model
-----
Two driven two-level systems, a and b, with opposite detunings:

    H_a = -δ σ⁺_a σ⁻_a + (Ω/2)(σ⁺_a + σ⁻_a),
    H_b = +δ σ⁺_b σ⁻_b + (Ω/2)(σ⁺_b + σ⁻_b).

The two systems undergo independent spontaneous emission,

    c_a = √γ σ⁻_a,
    c_b = √γ σ⁻_b.

The Hilbert-space basis is ordered as

    |ee⟩, |eg⟩, |ge⟩, |gg⟩.

The asymptotic quantum Fisher information (QFI) rate is obtained from the
dominant eigenvalue of a generalized Liouvillian. For two nearby parameter
values θ and θ + dθ, the generalized density matrix evolves according to

    dρ_{θ,θ+dθ}/dt = L_{θ,θ+dθ}[ρ_{θ,θ+dθ}],

where the Hamiltonian on the ket and bra sides is evaluated at the two
different parameter values.

If λ(θ, θ + dθ) denotes the dominant eigenvalue of the generalized
Liouvillian, the QFI rate is evaluated using the symmetric combination

    Ḟ_θ =
        -4 [λ(θ, θ + dθ) + λ(θ + dθ, θ)] / dθ².

The parameter to be estimated can be either the Rabi frequency Ω or the
detuning δ.

Numerical notes
---------------
- Sparse matrices are used to construct the generalized Liouvillian.
- The dominant eigenvalue is calculated using ARPACK.
- Independent detuning points are evaluated in parallel using `pmap`.
- Convergence with respect to the finite parameter shift `dp` and the
  eigensolver tolerance should be checked for production calculations.
"""


using SparseArrays
using LinearAlgebra
using Distributed
using Plots
using LaTeXStrings
using DataFrames
using CSV
using Arpack


# ==============================================================================
# 1. Hamiltonian
# ==============================================================================

"""
    compute_Hamiltonian(δ, Ω)

Construct the Hamiltonian of two driven two-level systems with opposite
detunings,

    H_a = -δ σ⁺σ⁻ + (Ω/2)(σ⁺ + σ⁻),
    H_b = +δ σ⁺σ⁻ + (Ω/2)(σ⁺ + σ⁻).

The total Hamiltonian is

    H = H_a ⊗ I + I ⊗ H_b.
"""
function compute_Hamiltonian(
    δ::Real,
    Ω::Real,
)
    σm = sparse(ComplexF64[
        0 0
        1 0
    ])

    σp = σm'

    I_2 = sparse(I(2))

    H_a =
        -δ .* (σp * σm) .+
        (Ω / 2) .* (σp .+ σm)

    H_b =
         δ .* (σp * σm) .+
        (Ω / 2) .* (σp .+ σm)

    return (
        kron(H_a, I_2) +
        kron(I_2, H_b)
    )
end


# ==============================================================================
# 2. Generalized Liouvillian
# ==============================================================================

"""
    compute_liouvillian(δ0, Ω0, δ1, Ω1, γ)

Construct the generalized Liouvillian associated with two parameter values,

    θ₀ = (δ₀, Ω₀),
    θ₁ = (δ₁, Ω₁).

The coherent contribution is

    -i [H(θ₀) ρ - ρ H(θ₁)],

while the dissipative channels are identical for the two parameter values.

Using the vectorization convention

    vec(A ρ B) = (Bᵀ ⊗ A) vec(ρ),

the generalized Liouvillian is represented as a sparse matrix.

For θ₀ = θ₁, the generalized evolution reduces to the ordinary Lindblad
master equation.
"""
function compute_liouvillian(
    δ0::Real,
    Ω0::Real,
    δ1::Real,
    Ω1::Real,
    γ::Real,
)
    σm = sparse(ComplexF64[
        0 0
        1 0
    ])

    I_2 = sparse(I(2))
    I_4 = sparse(I(4))

    # Hamiltonians evaluated at the two parameter values
    H0 = compute_Hamiltonian(
        δ0,
        Ω0,
    )

    H1 = compute_Hamiltonian(
        δ1,
        Ω1,
    )

    # Independent spontaneous-emission channels
    c_jump_1 =
        kron(
            sqrt(γ) .* σm,
            I_2,
        )

    c_jump_2 =
        kron(
            I_2,
            sqrt(γ) .* σm,
        )

    # Generalized coherent evolution:
    #
    #     -i [H(θ₀)ρ - ρH(θ₁)]
    L =
        -im .* (
            kron(I_4, H0) -
            kron(transpose(H1), I_4)
        )

    # Jump contribution from channel 1
    L +=
        kron(
            conj(c_jump_1),
            c_jump_1,
        )

    cdagc_1 =
        c_jump_1' * c_jump_1

    L +=
        -0.5 .* kron(
            I_4,
            cdagc_1,
        )

    L +=
        -0.5 .* kron(
            transpose(cdagc_1),
            I_4,
        )

    # Jump contribution from channel 2
    L +=
        kron(
            conj(c_jump_2),
            c_jump_2,
        )

    cdagc_2 =
        c_jump_2' * c_jump_2

    L +=
        -0.5 .* kron(
            I_4,
            cdagc_2,
        )

    L +=
        -0.5 .* kron(
            transpose(cdagc_2),
            I_4,
        )

    return L
end


# ==============================================================================
# 3. Dominant Liouvillian eigenvalue
# ==============================================================================

"""
    compute_dominant_eigenvalue(L)

Calculate the eigenvalue of the generalized Liouvillian closest to zero.

For small parameter displacements, this eigenvalue continuously connects to
the zero eigenvalue of the physical Liouvillian and determines the long-time
overlap decay between output states corresponding to nearby parameter values.

A shift-invert ARPACK calculation is used to target the eigenvalue near zero.
"""
function compute_dominant_eigenvalue(
    L::SparseMatrixCSC,
)
    eigenvalues, _ = eigs(
        L;
        nev = 1,
        sigma = 1.0e-4,
        which = :LM,
        tol = 1.0e-10,
        maxiter = 2000,
    )

    return eigenvalues[1]
end


# ==============================================================================
# 4. Asymptotic QFI rate
# ==============================================================================

"""
    compute_qfi_rate(δ, Ω, γ, dp, param)

Calculate the asymptotic QFI rate using a finite parameter displacement.

For `param == :Omega`,

    Ω → Ω + dp,

while for `param == :delta`,

    δ → δ + dp.

The generalized Liouvillian is evaluated in both directions,

    L(θ, θ + dθ)

and

    L(θ + dθ, θ),

and the QFI rate is obtained from

    Ḟ_θ =
        -4 [λ(θ, θ + dθ) + λ(θ + dθ, θ)] / dθ².

The returned value is real up to numerical eigensolver precision.
"""
function compute_qfi_rate(
    δ::Real,
    Ω::Real,
    γ::Real,
    dp::Real,
    param::Symbol,
)
    if dp <= 0
        throw(ArgumentError(
            "dp must be positive."
        ))
    end

    if !(param in (:Omega, :delta))
        throw(ArgumentError(
            "param must be :Omega or :delta, got $param"
        ))
    end

    # Apply a small parameter displacement
    if param == :Omega

        δ_shift = δ
        Ω_shift = Ω + dp

    else  # param == :delta

        δ_shift = δ + dp
        Ω_shift = Ω
    end

    # Forward generalized Liouvillian:
    #     θ -> θ + dθ
    L_forward = compute_liouvillian(
        δ,
        Ω,
        δ_shift,
        Ω_shift,
        γ,
    )

    λ_forward =
        compute_dominant_eigenvalue(
            L_forward,
        )

    # Reverse generalized Liouvillian:
    #     θ + dθ -> θ
    L_reverse = compute_liouvillian(
        δ_shift,
        Ω_shift,
        δ,
        Ω,
        γ,
    )

    λ_reverse =
        compute_dominant_eigenvalue(
            L_reverse,
        )

    # Asymptotic QFI rate
    qfi_rate =
        -4 .* (
            λ_forward +
            λ_reverse
        ) ./ dp^2

    return real(qfi_rate)
end


# ==============================================================================
# 5. Visualization
# ==============================================================================

"""
    save_plot(
        detuning_values,
        qfi_rates,
        param,
        imgfile,
    )

Plot the asymptotic QFI rate as a function of the detuning and save the
figure as a PNG file.
"""
function save_plot(
    detuning_values,
    qfi_rates,
    param::Symbol,
    imgfile::String,
)
    p = plot(
        xlabel = L"\delta/\gamma",
        ylabel =
            param == :Omega ?
            L"\dot{I}_{\Omega}" :
            L"\dot{I}_{\delta}",
        legend = false,
        size = (800, 600),
        guidefont = font(20),
        tickfont = font(18),
        legendfont = font(18),
        titlefont = font(20),
        framestyle = :box,
        grid = false,
    )

    plot!(
        detuning_values,
        qfi_rates;
        seriestype = :scatter,
        markersize = 6,
        color = :blue,
    )

    savefig(
        p,
        imgfile,
    )

    println(
        "Figure saved to: $imgfile"
    )

    return p
end


# ==============================================================================
# 6. Parallel worker setup
# ==============================================================================

"""
    setup_workers(nworkers_req=0)

Initialize worker processes for parallel calculations.

If `nworkers_req == 0`, the number of workers is chosen automatically from
the available CPU threads. Otherwise, `nworkers_req` specifies the desired
number of worker processes.

The current script is loaded on all workers so that the required functions
are available during `pmap`.
"""
function setup_workers(
    nworkers_req::Int = 0,
)
    if nworkers_req < 0
        throw(ArgumentError(
            "nworkers_req must be non-negative."
        ))
    end

    if nworkers() < 2

        n_add = if nworkers_req == 0
            max(1, Sys.CPU_THREADS - 2)
        else
            nworkers_req
        end

        addprocs(n_add)

        println(
            "Started $n_add worker process" *
            (n_add == 1 ? "." : "es.")
        )

    else

        println(
            "Using $(nworkers()) existing worker processes."
        )
    end

    # Make all function definitions available on every worker
    @everywhere workers() include(@__FILE__)

    println(
        "Total processes (including master): $(nprocs())"
    )

    return workers()
end


# ==============================================================================
# 7. Main calculation
# ==============================================================================

"""
    main(; kwargs...)

Calculate the asymptotic QFI rate over a range of detunings, save the
numerical results to a CSV file, and generate a figure.

Keyword arguments
-----------------
- `Ω`              : Rabi frequency.
- `γ`              : Spontaneous-emission rate.
- `dp`             : Finite parameter shift.
- `detuning_min`   : Minimum detuning in the scan.
- `detuning_max`   : Maximum detuning in the scan.
- `n_points`       : Number of detuning points.
- `param`          : Parameter to estimate, `:Omega` or `:delta`.
- `outdir`         : Output directory.
- `nworkers_req`   : Requested number of worker processes.
- `state_label`    : Label used in the output filenames.
"""
function main(;
    Ω::Float64 = 1.0,
    γ::Float64 = 1.0,
    dp::Float64 = 1.0e-5,
    detuning_min::Float64 = -4.0,
    detuning_max::Float64 = 4.0,
    n_points::Int = 200,
    param::Symbol = :delta,
    outdir::String = "figure",
    nworkers_req::Int = 0,
    state_label::String = "gg",
)
    # --------------------------------------------------------------------------
    # Input validation
    # --------------------------------------------------------------------------

    if γ < 0
        throw(ArgumentError(
            "γ must be non-negative."
        ))
    end

    if dp <= 0
        throw(ArgumentError(
            "dp must be positive."
        ))
    end

    if n_points < 2
        throw(ArgumentError(
            "n_points must be at least 2."
        ))
    end

    if detuning_max <= detuning_min
        throw(ArgumentError(
            "detuning_max must be larger than detuning_min."
        ))
    end

    if !(param in (:Omega, :delta))
        throw(ArgumentError(
            "param must be :Omega or :delta, got $param"
        ))
    end

    # --------------------------------------------------------------------------
    # Parallel workers
    # --------------------------------------------------------------------------

    setup_workers(nworkers_req)

    # --------------------------------------------------------------------------
    # Output files
    # --------------------------------------------------------------------------

    mkpath(outdir)

    param_str =
        param == :Omega ?
        "Omega" :
        "delta"

    file_prefix =
        "QFI_rate_$(state_label)_$(param_str)" *
        "_O$(Ω)_g$(γ)"

    csv_file = joinpath(
        outdir,
        "$(file_prefix).csv",
    )

    figure_file = joinpath(
        outdir,
        "$(file_prefix).png",
    )

    # --------------------------------------------------------------------------
    # Simulation summary
    # --------------------------------------------------------------------------

    println("="^70)
    println("Asymptotic QFI Rate of Two Driven Two-Level Systems")
    println("Method: Dominant eigenvalue of the generalized Liouvillian")
    println("Ω = $Ω")
    println("γ = $γ")
    println("Parameter to estimate: $param")
    println("Finite parameter shift: $dp")
    println(
        "Detuning range: " *
        "[$detuning_min, $detuning_max]"
    )
    println("Number of detuning points: $n_points")
    println("="^70)

    # --------------------------------------------------------------------------
    # Detuning scan
    # --------------------------------------------------------------------------

    detuning_values =
        collect(
            range(
                detuning_min,
                detuning_max;
                length = n_points,
            )
        )

    println(
        "\nCalculating QFI rate over " *
        "$n_points detuning points..."
    )

    qfi_rates = pmap(
        detuning_values,
    ) do δ

        compute_qfi_rate(
            δ,
            Ω,
            γ,
            dp,
            param,
        )
    end

    println(
        "QFI-rate calculation completed."
    )

    # --------------------------------------------------------------------------
    # Save numerical data
    # --------------------------------------------------------------------------

    data = DataFrame(
        delta = detuning_values,
        QFI_rate = qfi_rates,
    )

    CSV.write(
        csv_file,
        data,
    )

    println(
        "\nData saved to: $csv_file"
    )

    # --------------------------------------------------------------------------
    # Save figure
    # --------------------------------------------------------------------------

    save_plot(
        detuning_values,
        qfi_rates,
        param,
        figure_file,
    )

    return (
        detuning_values,
        qfi_rates,
    )
end


# ==============================================================================
# 8. Script entry point
# ==============================================================================
#
# Run directly with
#
#     julia two_tls_qfi_rate.jl
#
# Worker processes are created automatically. There is no need to start Julia
# with `julia -p N`.
#
# Set `nworkers_req = 0` to choose the number of workers automatically, or set
# it to a positive integer to request a specific number of worker processes.
# ==============================================================================

if myid() == 1

    # --------------------------------------------------------------------------
    # Calculation parameters
    # --------------------------------------------------------------------------

    Ω = 1.0
    γ = 1.0

    # Finite parameter shift.
    # Convergence with respect to dp should be checked for production results.
    dp = 1.0e-5

    # Detuning scan
    detuning_min = -4.0
    detuning_max = 4.0
    n_points = 20  # increase it

    # Parameter to estimate: :Omega or :delta
    param = :delta

    # 0: automatically choose workers
    # N > 0: request N workers
    nworkers_req = 5   # choose appropriate value

    # Label used in output filenames
    state_label = "gg"

    # --------------------------------------------------------------------------
    # Run
    # --------------------------------------------------------------------------

    @time detuning_values, qfi_rates = main(
        Ω = Ω,
        γ = γ,
        dp = dp,
        detuning_min = detuning_min,
        detuning_max = detuning_max,
        n_points = n_points,
        param = param,
        outdir = "figure",
        nworkers_req = nworkers_req,
        state_label = state_label,
    )

    println(
        "\nCalculation completed successfully."
    )
end