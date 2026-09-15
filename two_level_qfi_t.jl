"""
Quantum Fisher Information for a Pair of Driven Two-Level Systems

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

The QFI is evaluated using a generalized density matrix ρ_{θ₀,θ₁}, whose
evolution contains H(θ₀) on the ket side and H(θ₁) on the bra side.

Two finite-difference estimators are evaluated:

    1. Total QFI from the overlap of the full system-output state.
    2. Output-field QFI from the trace norm of the generalized density matrix.

The parameter to be estimated can be either the Rabi frequency Ω or the
detuning δ.

Numerical notes
---------------
- Sparse matrices are used to construct the generalized Liouvillian.
- The generalized density matrix is propagated by matrix exponentiation.
- Its trace norm is evaluated through singular values rather than an explicit
  matrix square root for improved numerical stability.
- Independent time points are evaluated in parallel using `pmap`.
- Convergence with respect to the finite parameter shift `dp` should be
  checked for production calculations.
"""


using SparseArrays
using LinearAlgebra
using Distributed
using Plots
using LaTeXStrings
using DataFrames
using CSV


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

Construct the generalized Liouvillian governing the off-diagonal density
matrix ρ_{θ₀,θ₁} associated with two different parameter values.

The coherent contribution is

    -i [H(θ₀)ρ - ρH(θ₁)],

while the dissipative channels are identical for the two parameter values.

Using the vectorization convention

    vec(AρB) = (Bᵀ ⊗ A) vec(ρ),

the generalized evolution is represented as a sparse matrix.

For θ₀ = θ₁, the generalized Liouvillian reduces to the ordinary Lindblad
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
# 3. Generalized evolution
# ==============================================================================

"""
    compute_rho_t_trace(rho0, L, t)

Propagate the generalized density matrix to time `t` and return

    overlap    = Tr[ρ(t)],
    trace_norm = ||ρ(t)||₁.

The trace norm is evaluated as

    ||ρ||₁ = Σᵢ sᵢ(ρ),

where sᵢ are the singular values of ρ.

The generalized density matrix is converted to a dense matrix before the
singular-value decomposition. For the present two-TLS system, ρ is only a
4 × 4 matrix, so this conversion is inexpensive.

This avoids explicitly evaluating

    Tr[sqrt(ρρ†)],

which is less numerically stable near the physical-density-matrix limit.
"""
function compute_rho_t_trace(
    rho0::SparseMatrixCSC,
    L::SparseMatrixCSC,
    t::Real,
)
    n_rows, n_cols = size(rho0)

    rho0_vec = reshape(
        rho0,
        :,
    )

    rho_t_vec =
        exp(Matrix(L) .* t) *
        rho0_vec

    # Convert the propagated generalized density matrix to a dense matrix
    rho_t = Matrix(
        reshape(
            rho_t_vec,
            n_rows,
            n_cols,
        )
    )

    # Generalized overlap
    overlap = tr(rho_t)

    # Trace norm ||ρ||₁ = Σᵢ sᵢ(ρ)
    trace_norm =
        sum(svdvals(rho_t))

    return (
        overlap,
        real(trace_norm),
    )
end


# ==============================================================================
# 4. Quantum Fisher information
# ==============================================================================

"""
    compute_qfi(δ, Ω, γ, rho0, t, dp, param)

Calculate finite-difference QFI estimators at time `t`.

For `param == :Omega`,

    Ω → Ω + dp,

while for `param == :delta`,

    δ → δ + dp.

Two quantities are returned:

    QFI_total
        Total QFI obtained from the generalized overlap of the full
        system-output state.

    QFI_field
        QFI contained in the output field, obtained from the trace norm of
        the generalized density matrix.

The generalized evolution is evaluated in both directions,

    θ → θ + dθ

and

    θ + dθ → θ,

to form the symmetric finite-difference estimator for the total QFI.
"""
function compute_qfi(
    δ::Real,
    Ω::Real,
    γ::Real,
    rho0::SparseMatrixCSC,
    t::Real,
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

    # Forward generalized evolution:
    #     θ -> θ + dθ
    L_forward = compute_liouvillian(
        δ,
        Ω,
        δ_shift,
        Ω_shift,
        γ,
    )

    overlap_forward,
    trace_norm = compute_rho_t_trace(
        rho0,
        L_forward,
        t,
    )

    # Reverse generalized evolution:
    #     θ + dθ -> θ
    L_reverse = compute_liouvillian(
        δ_shift,
        Ω_shift,
        δ,
        Ω,
        γ,
    )

    overlap_reverse, _ =
        compute_rho_t_trace(
            rho0,
            L_reverse,
            t,
        )

    # Total QFI
    QFI_total = real(
        -4 .* (
            log(overlap_forward) +
            log(overlap_reverse)
        ) ./ dp^2
    )

    # Output-field QFI
    QFI_field = real(
        8 .* (
            1 - trace_norm
        ) ./ dp^2
    )

    return (
        QFI_total,
        QFI_field,
    )
end


# ==============================================================================
# 5. Visualization
# ==============================================================================

"""
    save_plot(
        t,
        QFI_total,
        QFI_field,
        imgfile,
    )

Plot the total and output-field QFI as functions of time and save the figure
as a PNG file.
"""
function save_plot(
    t,
    QFI_total,
    QFI_field,
    imgfile::String,
)
    p = plot(
        xlabel = L"\gamma t",
        ylabel = L"QFI",
        legend = :topleft,
        size = (800, 600),
        guidefont = font(20),
        tickfont = font(18),
        legendfont = font(18),
        titlefont = font(20),
        framestyle = :box,
        grid = false,
    )

    plot!(
        p,
        t,
        QFI_total;
        seriestype = :line,
        label = "Total QFI",
        linewidth = 2.5,
        color = :black,
    )

    plot!(
        p,
        t,
        QFI_field;
        seriestype = :scatter,
        marker = :star5,
        label = "Field QFI",
        markersize = 7,
        color = :red,
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
# 6. Main calculation
# ==============================================================================

"""
    main(; kwargs...)

Calculate the total and output-field QFI as functions of time.

The function automatically initializes the requested worker processes,
evaluates independent time points in parallel, saves the numerical data to a
CSV file, and generates a comparison figure.

Keyword arguments
-----------------
- `δ`              : Detuning magnitude.
- `Ω`              : Rabi frequency.
- `γ`              : Spontaneous-emission rate.
- `dp`             : Finite parameter shift.
- `Tfinal`         : Maximum evolution time.
- `n_points`       : Number of sampled time points.
- `param`          : Parameter to estimate, `:Omega` or `:delta`.
- `outdir`         : Output directory.
- `nworkers_req`   : Requested number of worker processes.
- `state_label`    : Label used in output filenames.
- `ψ0`             : Initial pure state. Defaults to |gg⟩.
"""
function main(;
    δ::Float64 = 0.5,
    Ω::Float64 = 1.0,
    γ::Float64 = 1.0,
    dp::Float64 = 1.0e-5,
    Tfinal::Float64 = 100.0,
    n_points::Int = 20,
    param::Symbol = :Omega,
    outdir::String = "figure",
    nworkers_req::Int = 0,
    state_label::String = "gg",
    ψ0::Union{
        Vector{ComplexF64},
        Nothing,
    } = nothing,
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

    if Tfinal <= 0
        throw(ArgumentError(
            "Tfinal must be positive."
        ))
    end

    if n_points < 2
        throw(ArgumentError(
            "n_points must be at least 2."
        ))
    end

    if !(param in (:Omega, :delta))
        throw(ArgumentError(
            "param must be :Omega or :delta, got $param"
        ))
    end

    if nworkers_req < 0
        throw(ArgumentError(
            "nworkers_req must be non-negative."
        ))
    end

    # --------------------------------------------------------------------------
    # Parallel workers
    # --------------------------------------------------------------------------

    n_target =
        nworkers_req == 0 ?
        max(1, Sys.CPU_THREADS - 1) :
        nworkers_req

    n_current = length(workers())

    if n_current < n_target

        n_add = n_target - n_current

        addprocs(n_add)

        println(
            "Added $n_add worker process" *
            (n_add == 1 ? "." : "es.")
        )

    else

        println(
            "Using $n_current existing worker processes."
        )
    end

    # Make all function definitions available on every worker
    @everywhere workers() include(@__FILE__)

    println(
        "Total processes (including master): $(nprocs())"
    )

    # --------------------------------------------------------------------------
    # Initial state
    # --------------------------------------------------------------------------

    if ψ0 === nothing
        ψ0 = ComplexF64[
            0,
            0,
            0,
            1,
        ]
    end

    if length(ψ0) != 4
        throw(ArgumentError(
            "ψ0 must contain four amplitudes in the " *
            "|ee⟩, |eg⟩, |ge⟩, |gg⟩ basis."
        ))
    end

    ψ0 = ψ0 ./ norm(ψ0)

    rho0 =
        sparse(
            ψ0 * ψ0'
        )

    # --------------------------------------------------------------------------
    # Output files
    # --------------------------------------------------------------------------

    mkpath(outdir)

    param_str =
        param == :Omega ?
        "Omega" :
        "delta"

    file_prefix =
        "QFI_$(state_label)_$(param_str)" *
        "_T$(Tfinal)_d$(δ)_O$(Ω)_g$(γ)"

    csv_file = joinpath(
        outdir,
        "$(file_prefix).csv",
    )

    figure_file = joinpath(
        outdir,
        "$(file_prefix).png",
    )

    # --------------------------------------------------------------------------
    # Calculation summary
    # --------------------------------------------------------------------------

    println("="^70)
    println("Quantum Fisher Information of Two Driven Two-Level Systems")
    println("Method: Generalized-density-matrix evolution")
    println("δ = $δ")
    println("Ω = $Ω")
    println("γ = $γ")
    println("Parameter to estimate: $param")
    println("Finite parameter shift: $dp")
    println("Final time: $Tfinal")
    println("Number of time points: $n_points")
    println("="^70)

    # --------------------------------------------------------------------------
    # Parallel QFI calculation
    # --------------------------------------------------------------------------

    t_values =
        collect(
            range(
                0.0,
                Tfinal;
                length = n_points,
            )
        )

    println(
        "\nCalculating QFI at $n_points time points..."
    )

    results = pmap(
        t_values,
    ) do t

        compute_qfi(
            δ,
            Ω,
            γ,
            rho0,
            t,
            dp,
            param,
        )
    end

    QFI_total =
        Float64[
            result[1]
            for result in results
        ]

    QFI_field =
        Float64[
            result[2]
            for result in results
        ]

    println(
        "QFI calculation completed."
    )

    println(
        "Final total QFI: $(QFI_total[end])"
    )

    println(
        "Final field QFI: $(QFI_field[end])"
    )

    # --------------------------------------------------------------------------
    # Save numerical data
    # --------------------------------------------------------------------------

    data = DataFrame(
        time = t_values,
        QFI_total = QFI_total,
        QFI_field = QFI_field,
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
        t_values,
        QFI_total,
        QFI_field,
        figure_file,
    )

    return (
        t_values,
        QFI_total,
        QFI_field,
    )
end


# ==============================================================================
# 7. Script entry point
# ==============================================================================
#
# Run directly with
#
#     julia two_tls_qfi.jl
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

    δ = 0.5
    Ω = 1.0
    γ = 1.0

    # Finite parameter shift.
    # Convergence with respect to dp should be checked for production results.
    dp = 1.0e-5

    Tfinal = 100.0

    # Example setting. Increase this value for smoother output curves.
    n_points = 20

    # Parameter to estimate: :Omega or :delta
    param = :Omega

    # 0: automatically choose workers
    # N > 0: request N workers
    nworkers_req = 5  # increase

    # --------------------------------------------------------------------------
    # Initial state |gg>
    # --------------------------------------------------------------------------

    state_label = "gg"

    ψ0 = ComplexF64[
        0,
        0,
        0,
        1,
    ]

    # --------------------------------------------------------------------------
    # Run
    # --------------------------------------------------------------------------

    @time (
        t_values,
        QFI_total,
        QFI_field,
    ) = main(
        δ = δ,
        Ω = Ω,
        γ = γ,
        dp = dp,
        Tfinal = Tfinal,
        n_points = n_points,
        param = param,
        outdir = "figure",
        nworkers_req = nworkers_req,
        state_label = state_label,
        ψ0 = ψ0,
    )

    println(
        "\nCalculation completed successfully."
    )
end