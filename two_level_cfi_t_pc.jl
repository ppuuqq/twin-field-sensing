"""
Monte Carlo Simulation of Classical Fisher Information
under Continuous Photon Counting

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

Two photon-counting measurement schemes are considered:

    Local:
        c_a, c_b

    Collective:
        (c_a + c_b)/√2,
        (c_a - c_b)/√2.

The classical Fisher information (CFI) is evaluated using a finite parameter
shift. A reference state and a parameter-shifted state are propagated along
the same measurement record.

Jump times are sampled using the exact no-jump evolution and determined by
binary search, avoiding a fixed time-step trajectory simulation.

Note
----
The current implementation assumes that the estimated parameter enters only
the Hamiltonian and does not modify the jump operators.
"""


using Distributed
using LinearAlgebra
using CairoMakie
using DataFrames
using CSV


# ==============================================================================
# 1. Operator definitions
# ==============================================================================

const σp_1 = ComplexF64[
    0 1
    0 0
]

const σm_1 = ComplexF64[
    0 0
    1 0
]

const I2 = Matrix{ComplexF64}(I, 2, 2)

# Tensor-product operators in the basis |ee⟩, |eg⟩, |ge⟩, |gg⟩
const σpa = kron(σp_1, I2)
const σma = kron(σm_1, I2)

const σpb = kron(I2, σp_1)
const σmb = kron(I2, σm_1)

const na = σpa * σma
const nb = σpb * σmb


"""
    build_operators(Ω, δ, γ; measurement=:local)

Construct the non-Hermitian effective Hamiltonian and photon-counting
jump operators.

The system Hamiltonian is

    H = H_a + H_b,

with

    H_a = -δ n_a + (Ω/2)(σ⁺_a + σ⁻_a),
    H_b = +δ n_b + (Ω/2)(σ⁺_b + σ⁻_b).

The effective Hamiltonian is

    H_eff = H - (i/2) Σₖ cₖ†cₖ.

Supported measurement schemes:

- `:local`
      c_a, c_b

- `:collective`
      (c_a + c_b)/√2,
      (c_a - c_b)/√2

Returns
-------
`H_eff, jump_ops`
"""
function build_operators(
    Ω::Float64,
    δ::Float64,
    γ::Float64;
    measurement::Symbol = :local,
)
    # System Hamiltonian
    H_a = -δ .* na .+ (Ω / 2) .* (σpa .+ σma)
    H_b =  δ .* nb .+ (Ω / 2) .* (σpb .+ σmb)

    H_sys = H_a .+ H_b

    # Local spontaneous-emission operators
    c_a = sqrt(γ) .* σma
    c_b = sqrt(γ) .* σmb

    # Measurement channels
    jump_ops = if measurement == :local
        [
            c_a,
            c_b,
        ]

    elseif measurement == :collective
        [
            (c_a + c_b) / sqrt(2),
            (c_a - c_b) / sqrt(2),
        ]

    else
        throw(ArgumentError(
            "measurement must be :local or :collective, got $measurement"
        ))
    end

    # Non-Hermitian no-jump Hamiltonian
    decay = sum(c' * c for c in jump_ops)
    H_eff = H_sys .- (im / 2) .* decay

    return H_eff, jump_ops
end


# ==============================================================================
# 2. Jump-time solver
# ==============================================================================

"""
    find_jump_time(H_eff, ψ, target_norm2, t_max)

Find the relative jump time `t_jump ∈ (0, t_max]` satisfying

    ||exp(-i H_eff t_jump) ψ||² = target_norm2.

The norm decreases monotonically during the no-jump evolution, so the
crossing time is determined by binary search.

If the target norm is not reached before `t_max`, the function returns
`t_max`.
"""
function find_jump_time(
    H_eff::Matrix{ComplexF64},
    ψ::AbstractVector{ComplexF64},
    target_norm2::Float64,
    t_max::Float64,
)
    norm2_start = real(dot(ψ, ψ))

    if target_norm2 >= norm2_start
        return 0.0
    end

    # Check whether a jump occurs before t_max
    ψ_end = exp(-im .* H_eff .* t_max) * ψ
    norm2_end = real(dot(ψ_end, ψ_end))

    if norm2_end > target_norm2
        return t_max
    end

    # Binary search
    t_low = 0.0
    t_high = t_max

    tol = 1e-12
    max_iter = 50

    for _ in 1:max_iter
        t_mid = (t_low + t_high) / 2

        ψ_mid = exp(-im .* H_eff .* t_mid) * ψ
        norm2_mid = real(dot(ψ_mid, ψ_mid))

        if abs(norm2_mid - target_norm2) < tol ||
           (t_high - t_low) < tol
            return t_mid
        end

        if norm2_mid > target_norm2
            t_low = t_mid
        else
            t_high = t_mid
        end
    end

    return (t_low + t_high) / 2
end


# ==============================================================================
# 3. Single quantum trajectory
# ==============================================================================

"""
    run_trajectory(
        ψ0,
        dp,
        H_eff_ref,
        H_eff_shift,
        jump_ops,
        t_save,
    )

Simulate a single photon-counting quantum trajectory using variable time
steps.

The reference state and the parameter-shifted state are propagated under

    H_eff(θ)

and

    H_eff(θ + dθ),

respectively, while sharing the same photon-counting record.

Jump times are determined from the no-jump survival probability of the
reference trajectory.

Returns
-------
A vector containing the trajectory contribution to the CFI at each time
in `t_save`.
"""
function run_trajectory(
    ψ0::Vector{ComplexF64},
    dp::Float64,
    H_eff_ref::Matrix{ComplexF64},
    H_eff_shift::Matrix{ComplexF64},
    jump_ops::Vector{Matrix{ComplexF64}},
    t_save::Vector{Float64},
)
    n_save = length(t_save)
    CFI_vals = zeros(Float64, n_save)

    # Reference and parameter-shifted states
    ψ_ref = copy(ψ0)
    ψ_shift = copy(ψ0)

    t_current = 0.0
    save_idx = 1

    # Draw the first no-jump survival threshold
    current_norm2 = real(dot(ψ_ref, ψ_ref))
    target_norm2 = rand() * current_norm2

    while save_idx <= n_save
        t_target = t_save[save_idx]
        dt_max = t_target - t_current

        # Determine whether a jump occurs before the next save time
        t_jump = find_jump_time(
            H_eff_ref,
            ψ_ref,
            target_norm2,
            dt_max,
        )

        if t_jump >= dt_max - 1e-14

            # ------------------------------------------------------------------
            # No jump before the next save time
            # ------------------------------------------------------------------

            dt = dt_max

            U_ref = exp(-im .* H_eff_ref .* dt)
            U_shift = exp(-im .* H_eff_shift .* dt)

            ψ_ref = U_ref * ψ_ref
            ψ_shift = U_shift * ψ_shift

            t_current = t_target

            # Finite-difference score for this measurement record
            norm2 = real(dot(ψ_ref, ψ_ref))

            overlap_ratio =
                2 * real(dot(ψ_shift, ψ_ref)) / norm2

            CFI_vals[save_idx] =
                ((overlap_ratio - 2) / dp)^2

            save_idx += 1

        else

            # ------------------------------------------------------------------
            # Photon-counting jump
            # ------------------------------------------------------------------

            dt = t_jump

            # Propagate both states to the jump time
            U_ref = exp(-im .* H_eff_ref .* dt)
            U_shift = exp(-im .* H_eff_shift .* dt)

            ψ_ref = U_ref * ψ_ref
            ψ_shift = U_shift * ψ_shift

            t_current += dt

            # Jump probabilities for the reference trajectory
            rates = [
                real(dot(c * ψ_ref, c * ψ_ref))
                for c in jump_ops
            ]

            total_rate = sum(rates)

            # Numerical safeguard for an anomalously small jump rate
            if total_rate < 1e-15
                norm2 = real(dot(ψ_ref, ψ_ref))

                if norm2 > 1e-14
                    norm_factor = sqrt(norm2)

                    ψ_ref ./= norm_factor
                    ψ_shift ./= norm_factor
                end

                target_norm2 = rand()
                continue
            end

            # Select the detected photon-counting channel
            r_choice = rand() * total_rate
            cumulative_rate = 0.0
            jump_index = length(jump_ops)

            for (k, rate) in enumerate(rates)
                cumulative_rate += rate

                if r_choice <= cumulative_rate
                    jump_index = k
                    break
                end
            end

            c_jump = jump_ops[jump_index]

            # Apply the same measurement outcome to both states
            ψ_ref_after = c_jump * ψ_ref
            ψ_shift_after = c_jump * ψ_shift

            norm_after =
                sqrt(real(dot(ψ_ref_after, ψ_ref_after)))

            if norm_after <= 1e-15
                error(
                    "Encountered a jump with an approximately zero norm."
                )
            end

            # Normalize with respect to the reference trajectory
            ψ_ref .= ψ_ref_after ./ norm_after
            ψ_shift .= ψ_shift_after ./ norm_after

            # Draw the survival threshold for the next no-jump segment
            target_norm2 = rand()
        end
    end

    return CFI_vals
end


# ==============================================================================
# 4. Classical Fisher information
# ==============================================================================

"""
    compute_CFI(; kwargs...) -> (t_save, CFI_mean)

Calculate the classical Fisher information by averaging over independent
photon-counting trajectories.

Keyword arguments
-----------------
- `Ω`           : Rabi frequency.
- `δ`           : Detuning magnitude.
- `γ`           : Spontaneous-emission rate.
- `dp`          : Finite parameter shift.
- `Tfinal`      : Total evolution time.
- `n_save`      : Number of uniformly spaced output times.
- `Ntraj`       : Number of Monte Carlo trajectories.
- `measurement` : `:local` or `:collective`.
- `param`       : Parameter to estimate, `:Omega` or `:delta`.
- `ψ0`          : Initial state. Defaults to |gg⟩.

Note
----
The estimated parameter is assumed to enter only the Hamiltonian.
"""
function compute_CFI(;
    Ω::Float64 = 1.0,
    δ::Float64 = 0.5,
    γ::Float64 = 1.0,
    dp::Float64 = 1.0e-5,
    Tfinal::Float64 = 8.0,
    n_save::Int = 200,
    Ntraj::Int = 1000,
    measurement::Symbol = :local,
    param::Symbol = :Omega,
    ψ0::Union{Vector{ComplexF64}, Nothing} = nothing,
)
    if dp <= 0
        throw(ArgumentError("dp must be positive."))
    end

    if Tfinal <= 0
        throw(ArgumentError("Tfinal must be positive."))
    end

    if n_save < 2
        throw(ArgumentError("n_save must be at least 2."))
    end

    if Ntraj < 1
        throw(ArgumentError("Ntraj must be positive."))
    end

    if !(param in (:Omega, :delta))
        throw(ArgumentError(
            "param must be :Omega or :delta, got $param"
        ))
    end

    # Uniform output-time grid
    t_save = collect(range(0.0, Tfinal; length=n_save))

    # Default initial state |gg⟩
    if ψ0 === nothing
        ψ0 = ComplexF64[0, 0, 0, 1]
    end

    if length(ψ0) != 4
        throw(ArgumentError(
            "ψ0 must contain four amplitudes in the " *
            "|ee⟩, |eg⟩, |ge⟩, |gg⟩ basis."
        ))
    end

    # Reference operators
    H_eff_ref, jump_ops = build_operators(
        Ω,
        δ,
        γ;
        measurement = measurement,
    )

    # Parameter-shifted effective Hamiltonian
    if param == :Omega

        H_eff_shift, _ = build_operators(
            Ω + dp,
            δ,
            γ;
            measurement = measurement,
        )

    else  # param == :delta

        H_eff_shift, _ = build_operators(
            Ω,
            δ + dp,
            γ;
            measurement = measurement,
        )
    end

    # Parallel trajectory simulation
    results = pmap(1:Ntraj) do _
        run_trajectory(
            copy(ψ0),
            dp,
            H_eff_ref,
            H_eff_shift,
            jump_ops,
            t_save,
        )
    end

    # Ensemble average
    CFI_mean = sum(results) ./ Ntraj

    return t_save, CFI_mean
end


# ==============================================================================
# 5. Visualization
# ==============================================================================

"""
    save_plot(
        t,
        CFI_local,
        CFI_collective,
        Ω,
        δ,
        γ,
        Ntraj,
        param,
        imgfile,
    )

Plot the CFI for local and collective photon counting and save the figure
as a PNG file.
"""
function save_plot(
    t,
    CFI_local,
    CFI_collective,
    Ω,
    δ,
    γ,
    Ntraj,
    param,
    imgfile,
)
    param_str = param == :Omega ? "Ω" : "δ"

    fig = Figure(
        size = (800, 500),
        fontsize = 24,
    )

    ax = Axis(
        fig[1, 1];
        xlabel = L"$\gamma t$",
        ylabel = L"$CFI (photon counting)$",
        xgridvisible = true,
        ygridvisible = true,
        xgridstyle = :dash,
        ygridstyle = :dash,
        xgridcolor = (:black, 0.15),
        ygridcolor = (:black, 0.15),
    )

    lines!(
        ax,
        t,
        CFI_local;
        label = "Local measurement",
        color = :royalblue,
        linewidth = 2.5,
    )

    lines!(
        ax,
        t,
        CFI_collective;
        label = "Collective measurement",
        color = :crimson,
        linewidth = 2.5,
        linestyle = :dash,
    )

    axislegend(
        ax;
        position = :rb,
        framevisible = true,
    )

    save(
        imgfile,
        fig;
        px_per_unit = 2,
    )

    println("Figure saved to: $imgfile")

    return fig
end


# ==============================================================================
# 6. Parallel worker setup
# ==============================================================================

"""
    setup_workers(nworkers_req=0)

Initialize worker processes for parallel trajectory simulations.

If `nworkers_req == 0`, the number of workers is chosen automatically from
the available CPU threads. Otherwise, `nworkers_req` specifies the desired
number of workers.

The current script is loaded on all workers so that the trajectory functions
are available during `pmap`.
"""
function setup_workers(nworkers_req::Int = 0)

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
# 7. Main simulation
# ==============================================================================

"""
    main(; kwargs...)

Run both local and collective photon-counting simulations, save the CFI data
to a CSV file, and generate a comparison figure.
"""
function main(;
    Ω::Float64 = 1.0,
    δ::Float64 = 0.5,
    γ::Float64 = 1.0,
    dp::Float64 = 1.0e-5,
    Tfinal::Float64 = 8.0,
    Ntraj::Int = 2000,
    n_save::Int = 200,
    param::Symbol = :Omega,
    outdir::String = "figure",
    nworkers_req::Int = 0,
    state_label::String = "gg",
    ψ0::Union{Vector{ComplexF64}, Nothing} = nothing,
)
    # --------------------------------------------------------------------------
    # Parallel workers
    # --------------------------------------------------------------------------

    setup_workers(nworkers_req)

    # --------------------------------------------------------------------------
    # Output files
    # --------------------------------------------------------------------------

    mkpath(outdir)

    param_str = param == :Omega ? "Omega" : "delta"

    file_prefix =
        "CFI_PC_$(state_label)_$(param_str)" *
        "_N$(Ntraj)_T$(Tfinal)_d$(δ)_O$(Ω)_g$(γ)"

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
    println("Monte Carlo Simulation of Continuous-Measurement CFI")
    println("Method: Variable-time-step quantum trajectories")
    println("Ω = $Ω")
    println("δ = $δ")
    println("γ = $γ")
    println("Parameter to estimate: $param")
    println("Finite parameter shift: $dp")
    println("Number of trajectories: $Ntraj")
    println("Final time: $Tfinal")
    println("="^70)

    # --------------------------------------------------------------------------
    # Local photon counting
    # --------------------------------------------------------------------------

    println("\n[1/2] Running local photon counting...")

    t_local, CFI_local = compute_CFI(
        Ω = Ω,
        δ = δ,
        γ = γ,
        dp = dp,
        Tfinal = Tfinal,
        n_save = n_save,
        Ntraj = Ntraj,
        measurement = :local,
        param = param,
        ψ0 = ψ0,
    )

    println(
        "Local measurement completed. " *
        "CFI(T) = $(CFI_local[end])"
    )

    # --------------------------------------------------------------------------
    # Collective photon counting
    # --------------------------------------------------------------------------

    println("\n[2/2] Running collective photon counting...")

    t_collective, CFI_collective = compute_CFI(
        Ω = Ω,
        δ = δ,
        γ = γ,
        dp = dp,
        Tfinal = Tfinal,
        n_save = n_save,
        Ntraj = Ntraj,
        measurement = :collective,
        param = param,
        ψ0 = ψ0,
    )

    println(
        "Collective measurement completed. " *
        "CFI(T) = $(CFI_collective[end])"
    )

    # --------------------------------------------------------------------------
    # Save numerical data
    # --------------------------------------------------------------------------

    data = DataFrame(
        t = t_local,
        CFI_local = CFI_local,
        CFI_collective = CFI_collective,
    )

    CSV.write(csv_file, data)

    println("\nData saved to: $csv_file")

    # --------------------------------------------------------------------------
    # Save figure
    # --------------------------------------------------------------------------

    save_plot(
        t_local,
        CFI_local,
        CFI_collective,
        Ω,
        δ,
        γ,
        Ntraj,
        param,
        figure_file,
    )

    return t_local, CFI_local, CFI_collective
end


# ==============================================================================
# 8. Script entry point
# ==============================================================================
#
# Run directly with
#
#     julia continuous_meas_CFI.jl
#
# Worker processes are created automatically. There is no need to start Julia
# with `julia -p N`.
#
# Set `nworkers_req = 0` to choose the number of workers automatically, or set
# it to a positive integer to request a specific number of worker processes.
# ==============================================================================

if myid() == 1

    # --------------------------------------------------------------------------
    # Simulation parameters
    # --------------------------------------------------------------------------

    Ω = 1.0
    δ = 0.5
    γ = 1.0

    # Finite-difference shift.
    # Convergence with respect to dp should be checked for production results.
    dp = 1.0e-5

    Tfinal = 100.0

    # Example settings. Increase these values for production calculations.
    Ntraj = 100
    n_save = 20

    # Parameter to estimate: :Omega or :delta
    param = :Omega

    # 0: automatically choose workers
    # N > 0: request N workers
    nworkers_req = 5  # choose appropriate value

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

    @time t, CFI_local, CFI_collective = main(
        Ω = Ω,
        δ = δ,
        γ = γ,
        dp = dp,
        Tfinal = Tfinal,
        Ntraj = Ntraj,
        n_save = n_save,
        param = param,
        outdir = "figure",
        nworkers_req = nworkers_req,
        state_label = state_label,
        ψ0 = ψ0,
    )

    println("\nSimulation completed successfully.")
end