# References:
#
# - https://lammps.sandia.gov/doc/Howto_library.htm
# - https://lammps.sandia.gov/doc/Python_library.html
#
# Some wrapper methods were generated with help of Clang.jl

"""
Julia wrapper around `liblammps`.
"""
module LAMMPSWrapper


# Dependencies
using Base: cconvert

using Reexport
@reexport using FilePathsBase
@reexport using MPI

# JLL dependencies
using LAMMPS_jll


const LAMMPSPtr = Ptr{Cvoid}

"List of available LAMMPS packages"
const PACKAGES = String[]


# TODO: Design a way to check overlap of MPI communicators for different
# LAMMPS instances.
"""
    LAMMPSObject

Wrapper around LAMMPS objects.

# Fields
- `args::Vector{String}`: LAMMPS command-line arguments.
- `comm::MPI.Comm`: MPI communicator.
- `ptr::LAMMPSPtr`: Pointer to the running instance.
"""
mutable struct LAMMPSObject
    args::Vector{String}
    comm::MPI.Comm
    ptr::LAMMPSPtr

    @doc """
        LAMMPSObject(; args = String[], comm = MPI.COMM_WORLD)

    Creates a LAMMPS instance.

    # Keywords
    - `args::Vector{String} = String[]`: LAMMPS command-line arguments for this instance.
    - `comm::MPI.Comm = MPI.COMM_WORLD`: MPI communicator.
    """
    function LAMMPSObject(; args = String[], comm = MPI.COMM_WORLD)
        ptrref = Ref{LAMMPSPtr}(C_NULL)

        ccall(
            (:lammps_open, liblammps),
            Cvoid,
            (Cint, Ref{Cstring}, MPI.Comm, Ptr{LAMMPSPtr}),
            length(args), args, comm, ptrref
        )

        o = new(args, comm, ptrref[])
        finalizer(unset!, o)
        return o
    end
end

"""
    initialized(o::LAMMPSObject) :: Bool

Returns true if `o` has been instantiated from `liblammps`.
"""
initialized(o::LAMMPSObject) = o.ptr != C_NULL

"""
    check_state(o::LAMMPSObject)

Throws is instance `o` is currently inactive.
"""
function check_state(o::LAMMPSObject)
    if !initialized(o)
        error("LAMMPS instance is inactive")
    end
    return nothing
end

"""
    reset!(o::LAMMPSObject; args = o.args, comm = o.comm)

Restarts an inactive LAMMPS instance `o`.

# Arguments
- `o::LAMMPSObject`: Previously defined, but currently inactive, LAMMPS instance.

# Keywords
- `args::Vector{String} = o.args`: LAMMPS command-line arguments for this instance.
- `comm::MPI.Comm = o.comm`: MPI communicator.
"""
function reset!(o::LAMMPSObject; args = o.args, comm = o.comm)
    if initialized(o)
        error("cannot reset active LAMMPS objects")
    end

    ptrref = Ref{LAMMPSPtr}(C_NULL)

    ccall(
        (:lammps_open, liblammps),
        Cvoid,
        (Cint, Ref{Cstring}, MPI.Comm, Ptr{LAMMPSPtr}),
        length(args), args, comm, ptrref
    )

    o.args = args
    o.comm = comm
    o.ptr = ptrref[]
    return o
end

"""
    unset!(o::LAMMPSObject)

Closes LAMMPS instance `o` from `liblammps` and incativates it.
"""
function unset!(o::LAMMPSObject)
    if initialized(o)
        ccall((:lammps_close, liblammps), Cvoid, (LAMMPSPtr,), o.ptr)
        # Clean the pointer address so we can check safely before running any
        # command on a given instance
        o.ptr = C_NULL
    end
    return nothing
end

"""
    version(o::LAMMPSObject) :: Int

Returns the version of the underlying LAMMPS library as an integer with format
`YYYYMMDD`.
"""
function version(o::LAMMPSObject)
    check_state(o)
    return Int(ccall((:lammps_version, liblammps), Cint, (LAMMPSPtr,), o.ptr))
end

"""
    load!(o::LAMMPSObject, filename::AbstractPath)

Runs the LAMMPS input file `filename` on instance `o`.
"""
function load!(o::LAMMPSObject, path::AbstractPath)
    check_state(o)
    if !isfile(path)
        throw(ArgumentError("$filename does not exist or is not a file"))
    end
    ccall(
        (:lammps_file, liblammps),
        Cvoid,
        (LAMMPSPtr, Cstring),
        o.ptr, last(path.segments)
    )
    return nothing
end
"""
    load!(o::LAMMPSObject, cmd::String)

Runs LAMMPS command `cmd` on instance `o`.
"""
function load!(o::LAMMPSObject, cmd::String)
    check_state(o)
    if ispath(cmd)
        throw(ArgumentError("for running input files call `run(o, p\"$cmd\")` instead"))
    elseif '\n' in cmd
        # There is no point in supporting `:lammps_commands_string` from
        # `liblammps` if we already have support for `:lammps_commands_list`
        throw(ArgumentError(
            "for running multiple commands call `run(o, split($cmd, '\n'))` instead"
        ))
    end
    try
        ccall((:lammps_command, liblammps), Cstring, (LAMMPSPtr, Cstring), o.ptr, cmd)
    catch e
        throw(e)
    end
    return nothing
end
"""
    load!(o::LAMMPSObject, cmds::Vector{String})

Runs a list of LAMMPS commands `cmds` on instance `o`.
"""
function run(o::LAMMPSObject, cmds::Vector{String})
    check_state(o)
    ccall(
        (:lammps_commands_list, liblammps),
        Cvoid,
        (LAMMPSPtr, Cint, Ref{Cstring}),
        o.ptr, length(cmds), cmds
    )
    return nothing
end

"""
    sizeof(o::LAMMPSObject, typename::String)

Size of the LAMMPS data type `typename`.
"""
function sizeof(o::LAMMPSObject, typename::String)
    check_state(o)
    size = ccall(
        (:lammps_extract_setting, liblammps),
        Cint,
        (LAMMPSPtr, Cstring),
        o.ptr, typename
    )
    return Int(size)
end

"""
    get_box(o::LAMMPSObject) :: NamedTuple

Box information of LAMMPS instance `o`.

# Returns
NamedTuple with entries
- `o::Vector{Float64}`: min-coordinates corner.
- `hi::Vector{Float64}`: max-coordinates corner.
- `xy::Float64`: tilt factor for XY face.
- `yz::Float64`: tilt factor for YZ face.
- `xz::Float64`: tilt factor for XZ face.
- `periodicity::Vector{Int}`
- `box_change::Int`
"""
function get_box(o::LAMMPSObject)
    check_state(o)

    # See https://sourceforge.net/p/lammps/mailman/message/36726313 for
    # reference on the output structure to `lammps_extract_box`
    lo = Vector{Cdouble}(undef, 3)
    hi = Vector{Cdouble}(undef, 3)
    xy = Ref{Cdouble}(0)
    yz = Ref{Cdouble}(0)
    xz = Ref{Cdouble}(0)
    periodicity = Vector{Cint}(undef, 3)
    box_change = Ref{Cint}(0)

    ccall(
        (:lammps_extract_box, liblammps),
        Cvoid,
        (
            LAMMPSPtr, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble},
            Ref{Cdouble}, Ref{Cdouble}, Ref{Cint}, Ref{Cint}
        ),
        o.ptr, lo, hi, xy, yz, xz, periodicity, box_change
    )

    return (
        lo = lo,
        hi = hi,
        xy = xy[],
        yz = yz[],
        xz = xz[],
        periodicity = Vector{Int}(periodicity),
        box_change = Int(box_change[]),
    )
end

# TODO: Find out allowed `property` values and their respective data types
function get_global_ptr(o::LAMMPSObject, property::String)
    check_state(o)
    return ccall(
        (:lammps_extract_global, liblammps),
        Ptr{Cvoid},
        (LAMMPSPtr, Cstring),
        o.ptr, property
    )
end

# TODO: Find out allowed `property` values and their respective data types
function get_atom_ptr(o::LAMMPSObject, property::String)
    check_state(o)
    return ccall(
        (:lammps_extract_atom, liblammps),
        Ptr{Cvoid},
        (LAMMPSPtr, Cstring),
        o.ptr, property
    )
end

# TODO: Find out allowed `compute` values and their respective data types
function get_compute_ptr(o::LAMMPSObject, name, i, j)
    check_state(o)
    return ccall(
        (:lammps_extract_compute, liblammps),
        Ptr{Cvoid},
        (LAMMPSPtr, Cstring, Cint, Cint),
        o.ptr, name, i, j
    )
end

# TODO: Find out allowed `fix` values and their respective data types
function get_fix_ptr(o::LAMMPSObject, name, id, style, i, j)
    check_state(o)
    return ccall(
        (:lammps_extract_fix, liblammps),
        Ptr{Cvoid},
        (LAMMPSPtr, Cstring, Cint, Cint, Cint, Cint),
        o.ptr, name, id, style, i, j
    )
end

# TODO: Find out allowed `variable` values and their respective data types
function get_variable_ptr(o::LAMMPSObject, name, group)
    check_state(o)
    return ccall(
        (:lammps_extract_variable, liblammps),
        Ptr{Cvoid},
        (LAMMPSPtr, Cstring, Cstring),
        o.ptr, name, group
    )
end

const thermo_keywords = [
    "step", "elapsed", "elaplong", "dt", "time", "cpu", "tpcpu", "spcpu",
    "cpuremain", "part", "timeremain", "atoms", "temp", "press", "pe", "ke",
    "etotal", "enthalpy", "evdwl", "ecoul", "epair", "ebond", "eangle",
    "edihed", "eimp", "emol", "elong", "etail", "vol", "density", "lx", "ly",
    "lz", "xlo", "xhi", "ylo", "yhi", "zlo", "zhi", "xy", "xz", "yz", "xlat",
    "ylat", "zlat", "bonds", "angles", "dihedrals", "impropers", "pxx", "pyy",
    "pzz", "pxy", "pxz", "pyz", "fmax", "fnorm", "nbuild", "ndanger", "cella",
    "cellb", "cellc", "cellalpha",
]

const thermo_regexps = [
    r"^(c|f)_\w+(\[(\d*\*\d*|\d+)\])?$",
    r"^(c|f)_\w+\[\d+\]\[\d+\]",
    r"^v_\w+(\[\d+\])$",
]

"""
    get_thermo(o::LAMMPSObject, keyword::String) :: Float64

Returns the value of a thermo keyword. A non-exhaustive list of allowed
`keyword` values is stored in the `LAMMPS.thermo_keywords` variable. For a
detailed description of each valid `keyword`, refer to the [`thermo_style`]
section of the LAMMPS manual.

[`thermo_style`]: https://lammps.sandia.gov/doc/thermo_style.html
"""
function get_thermo(o::LAMMPSObject, keyword::String)
    check_state(o)
    is_valid_keyword = (
        keyword in thermo_keywords ||
        any(occursin(r, keyword) for r in thermo_regexps)
    )
    if !is_valid_keyword
        throw("invalid keyword")
    end
    value = ccall(
        (:lammps_get_thermo, liblammps),
        Cdouble,
        (LAMMPSPtr, Cstring),
        o.ptr,
        keyword
    )
    return Float64(value)
end

"""
    get_natoms(o::LAMMPSObject) :: Int

Returns the total number of atoms in the system.
"""
function get_natoms(o::LAMMPSObject)
    check_state(o)
    return Int(ccall((:lammps_get_natoms, liblammps), Cint, (LAMMPSPtr,), o.ptr))
end

"""
    set_variable!(o::LAMMPSObject, old::String, new::String)

Sets an existing string-style variable to a new string value.
"""
function set_variable!(o::LAMMPSObject, variable::String, name::String)
    check_state(o)
    flag = ccall(
        (:lammps_set_variable, liblammps),
        Cint,
        (LAMMPSPtr, Cstring, Cstring),
        o.ptr, variabl, name
    )
    if flag != 0
        throw("could not set variable $variable to $name")
    end
    return nothing
end

"""
    reset_box!(o::LAMMPSObject, lo::Vector{<:Real}, hi::Vector{<:Real},
               xy::Real, yz::Real, xz::Real)

Resets the size and shape of the simulation box.

# Arguments
- `o::LAMMPSObject`: LAMMPS instance.
- `lo::Vector{<:Real}`: min-coordinates corner.
- `hi::Vector{<:Real}`: max-coordinates corner.
- `xy::Real`: XY-plane tilt value.
- `yz::Real`: YZ-plane tilt value.
- `xz::Real`: XZ-plane tilt value.
"""
function reset_box!(
    o::LAMMPSObject, lo::Vector{<:Real}, hi::Vector{<:Real}, xy::Real, yz::Real, xz::Real
)
    check_state(o)
    @assert length(lo) == length(hi) == 3
    new_lo = cconvert(Vector{Cdouble}, lo)
    new_hi = cconvert(Vector{Cdouble}, hi)
    ccall(
        (:lammps_reset_box, liblammps),
        Cvoid,
        (LAMMPSPtr, Ref{Cdouble}, Ref{Cdouble}, Cdouble, Cdouble, Cdouble),
        o.ptr, new_lo, new_hi, xy, yz, xz
    )
    return nothing
end

#function gather_atoms(o::LAMMPSObject, arg2, arg3, arg4, arg5)
#    check_state(o)
#    ccall(
#        (:lammps_gather_atoms, liblammps),
#        Cvoid,
#        (LAMMPSPtr, Cstring, Cint, Cint, Ptr{Cvoid}),
#        o.ptr, arg2, arg3, arg4, arg5
#    )
#end

#function gather_atoms_concat(o::LAMMPSObject, arg2, arg3, arg4, arg5)
#    check_state(o)
#    ccall(
#        (:lammps_gather_atoms_concat, liblammps),
#        Cvoid,
#        (LAMMPSPtr, Cstring, Cint, Cint, Ptr{Cvoid}),
#        o.ptr, arg2, arg3, arg4, arg5
#    )
#end

#function gather_atoms_subset(o::LAMMPSObject, arg2, arg3, arg4, arg5, arg6, arg7)
#    check_state(o)
#    ccall(
#        (:lammps_gather_atoms_subset, liblammps),
#        Cvoid,
#        (LAMMPSPtr, Cstring, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cvoid}),
#        o.ptr, arg2, arg3, arg4, arg5, arg6, arg7
#    )
#end

#function scatter_atoms(o::LAMMPSObject, arg2, arg3, arg4, arg5)
#    check_state(o)
#    ccall(
#        (:lammps_scatter_atoms, liblammps),
#        Cvoid,
#        (LAMMPSPtr, Cstring, Cint, Cint, Ptr{Cvoid}),
#        o.ptr, arg2, arg3, arg4, arg5
#    )
#end

#function scatter_atoms_subset(o::LAMMPSObject, arg2, arg3, arg4, arg5, arg6, arg7)
#    check_state(o)
#    ccall(
#        (:lammps_scatter_atoms_subset, liblammps),
#        Cvoid,
#        (LAMMPSPtr, Cstring, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cvoid}),
#        o.ptr, arg2, arg3, arg4, arg5, arg6, arg7
#    )
#end

function set_callback!(o::LAMMPSObject, arg2, callback, arg4)
    check_state(o)
    ccall(
        (:lammps_set_fix_external_callback, liblammps),
        Cvoid,
        (LAMMPSPtr, Cstring, Ptr{Cvoid}, Ptr{Cvoid}),
        o.ptr, arg2, callback, arg4
    )
end

has_package(name::String) = uppercase(name) in PACKAGES

function has_gzip_support()
    return Bool(ccall((:lammps_config_has_gzip_support, liblammps), Cint, ()))
end

function has_png_support()
    return Bool(ccall((:lammps_config_has_png_support, liblammps), Cint, ()))
end

function has_jpeg_support()
    return Bool(ccall((:lammps_config_has_jpeg_support, liblammps), Cint, ()))
end

function has_ffmpeg_support()
    return Bool(ccall((:lammps_config_has_ffmpeg_support, liblammps), Cint, ()))
end

function has_exceptions()
    return Bool(ccall((:lammps_config_has_exceptions, liblammps), Cint, ()))
end

function pair_neighbor_list_index(
    o::LAMMPSObject, style::String; exact = true, nsub = 0, request = 0
)
    index = ccall(
        (:lammps_find_pair_neighlist, liblammps),
        Cint,
        (LAMMPSPtr, Cstring, Cint, Cint, Cint),
        o.ptr, style, exact, nsub, request
    )
    if index == -1
        throw("neighbor list index not found")
    end
    return Int(index)
end

function fix_neighbor_list_index(o::LAMMPSObject, id; request = 0)
    index = ccall(
        (:lammps_find_fix_neighlist, liblammps),
        Cint,
        (LAMMPSPtr, Cstring, Cint),
        o.ptr, id, request
    )
    if index == -1
        throw("neighbor list index not found")
    end
    return Int(index)
end

function compute_neighbor_list_index(o::LAMMPSObject, id; request = 0)
    index = ccall(
        (:lammps_find_compute_neighlist, liblammps),
        Cint,
        (LAMMPSPtr, Cstring, Cint),
        o.ptr, id, request
    )
    if index == -1
        throw("neighbor list index not found")
    end
    return Int(index)
end

function neighbor_list_size(o::LAMMPSObject, index)
    n = ccall(
        (:lammps_neighlist_num_elements, liblammps),
        Cint,
        (LAMMPSPtr, Cint),
        o.ptr, index
    )
    return Int(n)
end

function neighbor_list(o::LAMMPSObject, idx, element)
    check_state(o)
    nref = Ref{Cint}()
    idref = Ref{Cint}()
    ptrref = Ref{Ptr{Cint}}(C_NULL)
    ccall(
        (:lammps_neighlist_element_neighbors, liblammps),
        Cvoid,
        (LAMMPSPtr, Cint, Cint, Ref{Cint}, Ref{Cint}, Ref{Ptr{Cint}}),
        o.ptr, idx, element, idref, nref, ptrref
    )
    return idref[], unsafe_wrap(Array, ptrref[], nref[])
end

function create_atoms!(o::LAMMPSObject, ids, types, xs, vs, image, shrink)
    check_state(o)
    c_ids = cconvert(Vector{Cint}, ids)
    c_types = cconvert(Vector{Cint}, types)
    c_xs = cconvert(Vector{Cdouble}, xs)
    c_vs = cconvert(Vector{Cdouble}, vs)
    c_image = cconvert(Vector{Cdouble}, image)
    ccall(
        (:lammps_create_atoms, liblammps),
        Cvoid,
        (
            LAMMPSPtr, Cint, Ref{Cint}, Ref{Cint},
            Ref{Cdouble}, Ref{Cdouble}, Ref{Cint}, Cint
        ),
        o.ptr, length(c_ids), c_ids, c_types, c_xs, c_vs, c_image, shrink
    )
    return nothing
end


function __init__()
    if !MPI.Initialized()
        MPI.Init()
    end

    # Write the installed packages list to global variable `PACKAGES`
    package_count = ccall((:lammps_config_package_count, liblammps), Cint, ())
    buffer = Vector{UInt8}(undef, 32)
    buffer_ptr = pointer(buffer)
    empty!(PACKAGES)
    for i in 0:(package_count - 1)
        ccall(
            (:lammps_config_package_name, liblammps),
            Cint,
            (Cint, Cstring, Cint),
            i, buffer_ptr, length(buffer)
        )
        push!(PACKAGES, unsafe_string(buffer_ptr))
    end
end


end # module
