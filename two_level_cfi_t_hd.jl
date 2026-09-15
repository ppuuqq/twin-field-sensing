"""
Monte Carlo Simulation of Classical Fisher Information
under Continuous Homodyne Detection

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

Two homodyne measurement schemes are considered:

    Local:
        c_a, c_b

    Collective:
        (c_a + c_b)/√2,
        (c_a - c_b)/√2.

For each output channel, the monitored quadrature is defined by

    L_k = exp(-i φ_k) c_k,

where φ_k = 0 corresponds to the x quadrature and φ_k = π/2 corresponds
to the p quadrature.

The classical Fisher information (CFI) of the homodyne record is evaluated
using a tangent-state method. For a fixed measurement record,

    dy_k = ⟨L_k + L_k†⟩ dt + dW_k,

and the tangent state |ϕ⟩ gives the likelihood score

    score = ∂θ log p(record) = 2 Re⟨ψ|ϕ⟩.

The CFI is obtained from the trajectory average

    CFI(t) = E[score(t)^2].

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

Construct the system Hamiltonian and output operators for homodyne detection.

The system Hamiltonian is

    H = H_a + H_b,

with

    H_a = -δ n_a + (Ω/2)(σ⁺_a + σ⁻_a),
    H_b = +δ n_b + (Ω/2)(σ⁺_b + σ⁻_b).

Supported measurement schemes:

- `:local`
      c_a, c_b

- `:collective`
      (c_a + c_b)/√2,
      (c_a - c_b)/√2

Returns
-------
`H, c_ops`
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

    H = H_a .+ H_b

    # Local output operators
    c_a = sqrt(γ) .* σma
    c_b = sqrt(γ) .* σmb

    # Measurement channels
    c_ops = if measurement == :local
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

    return H, c_ops
end


"""
    dH_dparam(param)

Construct the derivative of the system Hamiltonian with respect to the
estimated parameter.

Supported parameters:

- `:Omega`
- `:delta`
"""
function dH_dparam(param::Symbol)

    if param == :Omega

        return 0.5 .* (
            σpa .+ σma .+
            σpb .+ σmb
        )

    elseif param == :delta

        return -na .+ nb

    else

        throw(ArgumentError(
            "param must be :Omega or :delta, got $param"
        ))
    end
end


"""
    apply_homodyne_phase(c_ops, homodyne_phase)

Apply the local-oscillator phase to each output channel,

    L_k = exp(-i φ_k) c_k.

A phase `φ = 0` measures the x quadrature, while `φ = π/2` measures the
p quadrature.

`homodyne_phase` may be either a scalar applied to all channels or a vector
containing one phase for each channel.
"""
function apply_homodyne_phase(
    c_ops::Vector{Matrix{ComplexF64}},
    homodyne_phase::Union{Float64, Vector{Float64}},
)
    n_channels = length(c_ops)

    phases = if homodyne_phase isa Float64
        fill(homodyne_phase, n_channels)
    else
        homodyne_phase
    end

    if length(phases) != n_channels
        throw(ArgumentError(
            "homodyne_phase must be a scalar or contain " *
            "$n_channels phases."
        ))
    end

    return [
        exp(-im * phases[k]) .* c_ops[k]
        for k in 1:n_channels
    ]
end


# ==============================================================================
# 2. Homodyne trajectory update
# ==============================================================================

"""
    homodyne_increment(ψ, L_ops, dt)

Generate the physical homodyne measurement increments

    dy_k = ⟨L_k + L_k†⟩ dt + √dt ξ_k,

where ξ_k are independent standard Gaussian random variables.
"""
function homodyne_increment(
    ψ::Vector{ComplexF64},
    L_ops::Vector{Matrix{ComplexF64}},
    dt::Float64,
)
    n_channels = length(L_ops)

    dy = zeros(Float64, n_channels)
    sqrt_dt = sqrt(dt)

    for k in 1:n_channels

        L = L_ops[k]

        mean_current =
            real(dot(ψ, (L .+ L') * ψ))

        dy[k] =
            mean_current * dt +
            sqrt_dt * randn()
    end

    return dy
end


"""
    homodyne_kraus(H, L_ops, dy, dt)

Construct the finite-time-step Kraus operator for the linear homodyne
stochastic Schrödinger equation.

The implementation includes the second-order Itō terms used in the
trajectory integration.
"""
function homodyne_kraus(
    H::Matrix{ComplexF64},
    L_ops::Vector{Matrix{ComplexF64}},
    dy::Vector{Float64},
    dt::Float64,
)
    dim = size(H, 1)

    M =
        Matrix{ComplexF64}(I, dim, dim) .-
        im .* H .* dt

    # Dissipative contribution
    decay = zeros(ComplexF64, dim, dim)

    for L in L_ops
        decay .+= L' * L
    end

    M .-= 0.5 .* decay .* dt

    # Single-channel stochastic contributions
    n_channels = length(L_ops)

    for k in 1:n_channels

        L = L_ops[k]

        M .+= L .* dy[k]

        M .+=
            0.5 .* (L * L) .*
            (dy[k]^2 - dt)
    end

    # Cross Itō terms between different measurement channels
    if n_channels >= 2

        for i in 1:(n_channels - 1)

            for j in (i + 1):n_channels

                M .+=
                    (L_ops[i] * L_ops[j]) .*
                    (dy[i] * dy[j])
            end
        end
    end

    return M
end


# ==============================================================================
# 3. Single quantum trajectory
# ==============================================================================

"""
    run_trajectory(
        ψ0,
        H,
        dH,
        L_ops,
        dt,
        n_steps,
        save_steps,
    )

Simulate a single continuous homodyne trajectory and evaluate its
contribution to the classical Fisher information.

The normalized conditional state |ψ⟩ and tangent state |ϕ⟩ are propagated
along the same homodyne measurement record. The measurement increments are
held fixed when differentiating with respect to the estimated parameter.

The likelihood score is

    score = 2 Re⟨ψ|ϕ⟩,

and the trajectory contribution to the CFI is

    score².

Returns
-------
A vector containing the trajectory contribution to the CFI at the requested
save times.
"""
function run_trajectory(
    ψ0::Vector{ComplexF64},
    H::Matrix{ComplexF64},
    dH::Matrix{ComplexF64},
    L_ops::Vector{Matrix{ComplexF64}},
    dt::Float64,
    n_steps::Int,
    save_steps::Vector{Int},
)
    n_save = length(save_steps)

    CFI_vals = zeros(Float64, n_save)

    # Conditional state
    ψ = copy(ψ0)
    ψ ./= norm(ψ)

    # Tangent state ∂|ψ̃⟩/∂θ used for the likelihood score
    ϕ = zeros(ComplexF64, length(ψ))

    # Since the estimated parameter enters only H,
    #
    #     ∂M/∂θ = -i (∂H/∂θ) dt.
    dM = -im .* dH .* dt

    save_idx = 1

    # CFI is zero at t = 0
    while save_idx <= n_save &&
          save_steps[save_idx] == 0

        CFI_vals[save_idx] = 0.0
        save_idx += 1
    end

    for step in 1:n_steps

        # Generate the physical homodyne record
        dy = homodyne_increment(
            ψ,
            L_ops,
            dt,
        )

        # Conditional evolution operator
        M = homodyne_kraus(
            H,
            L_ops,
            dy,
            dt,
        )

        new_ψ = M * ψ
        norm_ψ = norm(new_ψ)

        if norm_ψ < 1e-14
            error(
                "The trajectory norm became too small. " *
                "Try reducing dt."
            )
        end

        # Fixed-record tangent-state update
        ϕ =
            (dM * ψ .+ M * ϕ) ./
            norm_ψ

        ψ = new_ψ ./ norm_ψ

        # Likelihood score
        score =
            2.0 * real(dot(ψ, ϕ))

        # Store the CFI contribution at requested times
        while save_idx <= n_save &&
              save_steps[save_idx] == step

            CFI_vals[save_idx] = score^2
            save_idx += 1
        end
    end

    return CFI_vals
end


# ==============================================================================
# 4. Classical Fisher information
# ==============================================================================

"""
    make_save_schedule(Tfinal, dt, n_save)

Construct the time grid for the trajectory simulation and the corresponding
integer time-step indices.

The effective time step is adjusted slightly so that the final simulation
time is exactly `Tfinal`.

Returns
-------
`t_save, save_steps, dt_eff, n_steps`
"""
function make_save_schedule(
    Tfinal::Float64,
    dt::Float64,
    n_save::Int,
)
    n_steps = Int(round(Tfinal / dt))

    if n_steps < 1
        throw(ArgumentError(
            "Tfinal/dt must be at least 1."
        ))
    end

    dt_eff = Tfinal / n_steps

    raw_steps =
        round.(
            Int,
            collect(
                range(
                    0,
                    n_steps;
                    length = n_save,
                )
            )
        )

    save_steps = unique(raw_steps)
    t_save = save_steps .* dt_eff

    return t_save, save_steps, dt_eff, n_steps
end


"""
    compute_CFI(; kwargs...) -> (t_save, CFI_mean)

Calculate the classical Fisher information by averaging over independent
homodyne trajectories.

Keyword arguments
-----------------
- `Ω`              : Rabi frequency.
- `δ`              : Detuning magnitude.
- `γ`              : Spontaneous-emission rate.
- `Tfinal`         : Total evolution time.
- `dt`             : Homodyne integration time step.
- `n_save`         : Number of requested output times.
- `Ntraj`          : Number of Monte Carlo trajectories.
- `measurement`    : `:local` or `:collective`.
- `param`          : Parameter to estimate, `:Omega` or `:delta`.
- `homodyne_phase` : Scalar phase or one phase for each output channel.
- `ψ0`             : Initial state. Defaults to |gg⟩.

Note
----
The estimated parameter is assumed to enter only the Hamiltonian.
"""
function compute_CFI(;
    Ω::Float64 = 1.0,
    δ::Float64 = 0.5,
    γ::Float64 = 1.0,
    Tfinal::Float64 = 8.0,
    dt::Float64 = 1.0e-3,
    n_save::Int = 200,
    Ntraj::Int = 1000,
    measurement::Symbol = :local,
    param::Symbol = :Omega,
    homodyne_phase::Union{
        Float64,
        Vector{Float64},
    } = 0.0,
    ψ0::Union{
        Vector{ComplexF64},
        Nothing,
    } = nothing,
)
    # --------------------------------------------------------------------------
    # Input validation
    # --------------------------------------------------------------------------

    if Tfinal <= 0
        throw(ArgumentError(
            "Tfinal must be positive."
        ))
    end

    if dt <= 0
        throw(ArgumentError(
            "dt must be positive."
        ))
    end

    if n_save < 2
        throw(ArgumentError(
            "n_save must be at least 2."
        ))
    end

    if Ntraj < 1
        throw(ArgumentError(
            "Ntraj must be positive."
        ))
    end

    if !(param in (:Omega, :delta))
        throw(ArgumentError(
            "param must be :Omega or :delta, got $param"
        ))
    end

    # --------------------------------------------------------------------------
    # Time grid
    # --------------------------------------------------------------------------

    t_save,
    save_steps,
    dt_eff,
    n_steps = make_save_schedule(
        Tfinal,
        dt,
        n_save,
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

    # --------------------------------------------------------------------------
    # Hamiltonian and measurement operators
    # --------------------------------------------------------------------------

    H, c_ops = build_operators(
        Ω,
        δ,
        γ;
        measurement = measurement,
    )

    L_ops = apply_homodyne_phase(
        c_ops,
        homodyne_phase,
    )

    dH = dH_dparam(param)

    # --------------------------------------------------------------------------
    # Parallel trajectory simulation
    # --------------------------------------------------------------------------

    results = pmap(1:Ntraj) do _

        run_trajectory(
            copy(ψ0),
            H,
            dH,
            L_ops,
            dt_eff,
            n_steps,
            save_steps,
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
        series_labels,
        series_values,
        imgfile,
    )

Plot the CFI for all homodyne configurations and save the figure as a PNG
file.
"""
function save_plot(
    t::Vector{Float64},
    series_labels::Vector{String},
    series_values::Vector{Vector{Float64}},
    imgfile::String,
)
    fig = Figure(
        size = (900, 550),
        fontsize = 22,
    )

    ax = Axis(
        fig[1, 1];
        xlabel = L"$\gamma t$",
        ylabel = L"$CFI (homodyne)$",
        xgridvisible = true,
        ygridvisible = true,
        xgridstyle = :dash,
        ygridstyle = :dash,
        xgridcolor = (:black, 0.15),
        ygridcolor = (:black, 0.15),
    )

    line_styles = [
        :solid,
        :dash,
        :dot,
        :dashdot,
        :solid,
        :dash,
    ]

    for i in eachindex(series_values)

        lines!(
            ax,
            t,
            series_values[i];
            label = series_labels[i],
            linewidth = 2.5,
            linestyle =
                line_styles[
                    mod1(i, length(line_styles))
                ],
        )
    end

    axislegend(
        ax;
        position = :lt,
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
# 7. Main simulation
# ==============================================================================

"""
    main(; kwargs...)

Run the selected local and collective homodyne configurations, save all CFI
curves to a CSV file, and generate a comparison figure.

The default phase scan contains six configurations:

Local detection:
    (0, 0)
    (π/2, π/2)

Collective detection:
    (0, 0)
    (π/2, π/2)
    (0, π/2)
    (π/2, 0)
"""
function main(;
    Ω::Float64 = 1.0,
    δ::Float64 = 0.5,
    γ::Float64 = 1.0,
    Tfinal::Float64 = 8.0,
    dt::Float64 = 1.0e-3,
    Ntraj::Int = 2000,
    n_save::Int = 200,
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
        "CFI_HD_$(state_label)_$(param_str)" *
        "_N$(Ntraj)_T$(Tfinal)_dt$(dt)" *
        "_d$(δ)_O$(Ω)_g$(γ)"

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
    println("Monte Carlo Simulation of Continuous-Homodyne CFI")
    println("Method: Diffusive homodyne trajectories with tangent-state score")
    println("Ω = $Ω")
    println("δ = $δ")
    println("γ = $γ")
    println("Parameter to estimate: $param")
    println("Time step: $dt")
    println("Number of trajectories: $Ntraj")
    println("Final time: $Tfinal")
    println("="^70)

    # --------------------------------------------------------------------------
    # Homodyne configurations
    # --------------------------------------------------------------------------

    phase_runs = [
        (
            "collective_pi0",
            "Collective (π/2, 0)",
            :collective,
            [pi / 2, 0.0],
        ),

        (
            "collective_pipi",
            "Collective (π/2, π/2)",
            :collective,
            [pi / 2, pi / 2],
        ),

        (
            "local_pipi",
            "Local (π/2, π/2)",
            :local,
            [pi / 2, pi / 2],
        ),

        (
            "collective_00",
            "Collective (0, 0)",
            :collective,
            [0.0, 0.0],
        ),

        (
            "local_00",
            "Local (0, 0)",
            :local,
            [0.0, 0.0],
        ),
        (
            "collective_0pi",
            "Collective (0, π/2)",
            :collective,
            [0.0, pi / 2],
        ),
    ]

    t_reference = Float64[]

    series_keys = String[]
    series_labels = String[]

    series_values =
        Vector{Vector{Float64}}()

    # --------------------------------------------------------------------------
    # Run all homodyne configurations
    # --------------------------------------------------------------------------

    for (
        run_index,
        (
            key,
            label,
            measurement,
            phases,
        ),
    ) in enumerate(phase_runs)

        println(
            "\n[$run_index/$(length(phase_runs))] " *
            "Running $label..."
        )

        t_current, CFI_current = compute_CFI(
            Ω = Ω,
            δ = δ,
            γ = γ,
            Tfinal = Tfinal,
            dt = dt,
            n_save = n_save,
            Ntraj = Ntraj,
            measurement = measurement,
            param = param,
            homodyne_phase = phases,
            ψ0 = ψ0,
        )

        # Check that all configurations use the same output-time grid
        if isempty(t_reference)

            t_reference = t_current

        else

            if length(t_current) != length(t_reference)
                error(
                    "Saved time grids have different lengths."
                )
            end

            if maximum(
                abs.(t_current .- t_reference)
            ) >= 1e-12

                error(
                    "Saved time grids are inconsistent."
                )
            end
        end

        push!(
            series_keys,
            key,
        )

        push!(
            series_labels,
            label,
        )

        push!(
            series_values,
            CFI_current,
        )

        println(
            "$label completed. " *
            "CFI(T) = $(CFI_current[end])"
        )
    end

    # --------------------------------------------------------------------------
    # Save numerical data
    # --------------------------------------------------------------------------

    data = DataFrame(
        t = t_reference,
    )

    for (
        key,
        values,
    ) in zip(
        series_keys,
        series_values,
    )
        data[!, Symbol(key)] = values
    end

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
        t_reference,
        series_labels,
        series_values,
        figure_file,
    )

    return (
        t_reference,
        series_keys,
        series_values,
    )
end


# ==============================================================================
# 8. Script entry point
# ==============================================================================
#
# Run directly with
#
#     julia continuous_meas_CFI_HD.jl
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

    Tfinal = 100.0

    # Homodyne integration time step.
    # Convergence with respect to dt should be checked for production results.
    dt = 1.0e-2

    # Example settings. Increase Ntraj for production calculations.
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

    @time t, series_keys, series_values = main(
        Ω = Ω,
        δ = δ,
        γ = γ,
        Tfinal = Tfinal,
        dt = dt,
        Ntraj = Ntraj,
        n_save = n_save,
        param = param,
        outdir = "figure",
        nworkers_req = nworkers_req,
        state_label = state_label,
        ψ0 = ψ0,
    )

    println(
        "\nSimulation completed successfully."
    )
end