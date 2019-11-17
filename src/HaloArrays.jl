module HaloArrays

#=
Resources:

[1] https://pdfs.semanticscholar.org/9f38/75a9a639d2371708e64b15b9556fe1eb4039.pdf
[2] http://www.hector.ac.uk/cse/distributedcse/reports/cp2k/cp2k/node9.html

=#

using MPI
export HaloArray,
       nhalo,
       comm,
       origin

# Enums to specify the intent and location of a halo region. The combination
# of these two, plus the size of the array and the number of halo points
# along the N direction is sufficient to characterize uniquely the size
# and origin of the subarray corresponding to the halo regions
@enum Intent SEND RECV
@enum Region LEFT RIGHT CENTER ALL

"""
    swapregions(nprocesses::NTuple{N, Int}, isperiodic::NTuple{N, Bool}) where {N}

Construct an array of `N`-tuples of the enum type `Region` specifying the
regions where halo exchange between processors or within the same processor
should happen. The exchange should happen along directions `dim` where
`nprocesses[dim] > 1` and `isperiodic[dim]` is true.
"""
function swapregions(nprocesses::NTuple{N, Int},
                     isperiodic::NTuple{N, Bool}, economic::Bool) where {N}
    # vector of dimensions along which exchange must happen
    exdims = findall( (nprocesses .!= 1) .| isperiodic )

    # number of nprocesses over which exchange must happen
    ndims = length(exdims)

    # init
    regions = NTuple{N, Region}[]

    #
    AREA = economic == true ? CENTER : ALL

    if ndims ≥ 1
        push!(regions, ntuple(i -> i == exdims[1] ? LEFT  : AREA, N))
        push!(regions, ntuple(i -> i == exdims[1] ? RIGHT : AREA, N))
        if ndims ≥ 2
            push!(regions, ntuple(i -> i == exdims[1] ? AREA :
                                       i == exdims[2] ? LEFT   : AREA, N))
            push!(regions, ntuple(i -> i == exdims[1] ? AREA :
                                       i == exdims[2] ? RIGHT  : AREA, N))
            if ndims == 3
                push!(regions, ntuple(i -> i == exdims[1] ? AREA :
                                           i == exdims[2] ? AREA :
                                           i == exdims[3] ? LEFT  : AREA, N))
                push!(regions, ntuple(i -> i == exdims[1] ? AREA :
                                           i == exdims[2] ? AREA :
                                           i == exdims[3] ? RIGHT  : AREA, N))
            end
        end
    end

    # also add the opposite regions
    return regions
end


opposite(regions::NTuple{N, Region}) where {N} = _opposite.(regions)
_opposite(v::Region) = (v == LEFT  ? RIGHT :
                        v == RIGHT ? LEFT  : v)

"""
    subarray_size(region::NTuple{N, Region},
                    size::NTuple{N, Symbol},
                   nhalo::NTuple{N, Int}) where {N}

Calculate size of the subarray of an array of with internal region (excluding
halo points) of size `size` corresponding to the halo region specified by
`region`, where the array has `nhalo` halo points on the left and right
boundaries of the `N` cartesian directions.
"""
subarray_size(region::NTuple{N, Region},
                size::NTuple{N, Int},
               nhalo::NTuple{N, Int}) where {N} =
    ntuple(N) do i
        region[i] in (LEFT, RIGHT) && return nhalo[i]
        region[i] ==  CENTER       && return size[i]
        region[i] ==  ALL          && return size[i] + 2*nhalo[i]
    end

"""
    subarray_start(region::NTuple{N, Region},
                   intent::Intent,
                     size::NTuple{N, Int},
                    nhalo::NTuple{N, Int}) where {N}

Calculate the origin of the subarray of an array of with internal region (excluding
halo points) of size `size` corresponding to the halo region specified by `region`
and `intent`, where the array has `nhalo` halo points on the left and right
boundaries of the `N` cartesian directions.
"""
subarray_start(region::NTuple{N, Region},
               intent::Intent,
                 size::NTuple{N, Int},
                nhalo::NTuple{N, Int}) where {N} =
    ntuple(N) do i
        # shift the origin if we need to recv the data
        s = intent == RECV ? nhalo[i] : 0
        region[i] == LEFT    && return 1 + nhalo[i] - s
        region[i] == RIGHT   && return 1 +  size[i] + s
        region[i] == CENTER  && return 1 + nhalo[i]
        region[i] == ALL     && return 1
    end

# Type parameters are:
# T     : eltype
# N     : number of nprocesses
# NHALO : `N`-tuple of number of halo points in each cartesian direction. it is assumed
#          that the left and right boundaries have the same number of halo points
# SIZE  : the size of the internal region. This is hardcoded in the type signature
#         so that the compiler has access to it at compile time.
# A     : the underlying array. Typically just a standard julia array
struct HaloArray{T, N, NHALO, SIZE, A<:DenseArray{T, N}} <: DenseArray{T, N}
    data::A
    subarrays::Dict{Tuple{NTuple{N, Region}, Intent}, MPI.Datatype}
    haloregions::Vector{NTuple{N, Region}}
    comm::MPI.Comm
    """
        HaloArray{T}(comm::MPI.Comm,
                localsize::NTuple{N, Int},
                    nhalo::NTuple{N, Int}) where {T}

    Construct a `N` dimensional distributed array over the MPI communicator `comm`, 
    with halo regions along the `N` cartesian dimension of size specified by `nhalo`. 
    The local size of the array, i.e. excluding the halo points, is defined by 
    `localsize`. The global size of the distributed array is implicit by the 
    topology of the communicator, whic must have a cartesian topology of dimension `N`.
    """
    function HaloArray{T}(comm::MPI.Comm,
                     localsize::NTuple{N, Int},
                         nhalo::NTuple{N, Int}; economic::Bool=true) where {N, T}

        # i do not expect 4d arrays!
        N > 3 && throw(ArgumentError("invalid argument"))

        # nhalo should be nonzero
        minimum(nhalo) ≥ 0 ||
            throw(ArgumentError("invalid halo specification"))

        # check N is right
        MPI.Cartdim_get(comm) == N ||
            throw(ArgumentError("incompatible communicator cartesian topology"))

        # obtain cartesian nprocesses information and convert to N tuples
        stuff = MPI.Cart_get(comm, N)
        nprocesses = tuple( Int.(stuff[1])...)
        isperiodic = tuple(Bool.(stuff[2])...)

        # if nhalo == 0, nprocesses can't be periodic and have more then one proc
        for dim = 1:N
            if nhalo[dim] == 0 && (isperiodic[dim] || nprocesses[dim] != 1)
                throw(ArgumentError("invalid halo specification"))
            end
        end

        # size of actual array include internal region size plus twice the halo
        data = zeros(T, localsize .+ nhalo .+ nhalo)

        # construct a dict of the MPI subarray types for the halo regions that
        # need to be sent around. Note that a HaloArray might have other halo
        # regions that are not sent around to other processors, but are used
        # as ghost points for boundary calculations, e.g. at a wall.
        subarrays = Dict{Tuple{NTuple{N, Region}, Intent}, MPI.Datatype}()
        haloregions = swapregions(nprocesses, isperiodic, economic)
        for region in haloregions
            for intent in (SEND, RECV)
                subsize  = subarray_size(region, localsize, nhalo)
                substart = subarray_start(region, intent, localsize, nhalo)
                subarray = MPI.Type_Create_Subarray(N,
                                                    Cint[size(data)...],
                                                    Cint[subsize...],
                                                    Cint[(substart .- 1)...],
                                                    MPI.MPI_ORDER_FORTRAN, T)
                MPI.Type_Commit!(subarray)
                subarrays[(region, intent)] = subarray
            end
        end

        return new{T, N, nhalo,
                   localsize, typeof(data)}(data, subarrays, haloregions, comm)
    end

    """
        HaloArray{T}(comm::MPI.Comm,
               nprocesses::NTuple{N, Int},
               isperiodic::NTuple{N, Bool},
                localsize::NTuple{N, Int},
                    nhalo::NTuple{N, Int}) where {T}

    Create a Cartesian topology with number of processes `nprocesses` along each
    Cartesian directions with periodicity defined by `isperiodic`.
    """
    function HaloArray{T}(comm::MPI.Comm,
                    nprocesses::NTuple{N, Int},
                    isperiodic::NTuple{N, Bool},
                     localsize::NTuple{N, Int},
                         nhalo::NTuple{N, Int}; economic::Bool=true) where {N, T}
        # checks
        prod(nprocesses) == MPI.Comm_size(comm) ||
            throw(ArgumentError("incompatible nprocesses specification"))

        # construct new communicator
        new_comm = MPI.Cart_create(comm, N, Cint[nprocesses...],
                                            Cint[isperiodic...], false)

        return HaloArray{T}(new_comm, localsize, nhalo; economic=economic)
    end
end

"""
    nhalo(a::HaloArray{T, N}) where {T, N}

Return a `N`-tuple of `Int`s with the size of the halo region along the `N`
cartesian directions of the array. The size of the halo region is the same
on the "left" and "right" boundaries of the array.
"""
nhalo(a::HaloArray{T, N, NHALO}) where {T, N, NHALO} = NHALO

"""
    comm(a::HaloArray)

Return the MPI communicator underlying the data distribution of the array.
"""
comm(a::HaloArray) = a.comm

"""
    origin(a::HaloArray{T, N}) where {T< N}

A `N`-tuple of indices pointing to the element of the underlying parent array
corresponding to the element, e.g. for `N=3` to `a[1, 1, 1]`.
"""
origin(a::HaloArray) = nhalo(a) .+ 1

"""
    origin(a::AbstractArray{T, N}) where {T, N}

For generic `AbstractArray`s, return a `N`-tuple of `1`s.
"""
origin(a::AbstractArray{T, N}) where {T, N} = ntuple(i->1, N)


# array interface
Base.parent(a::HaloArray) = a.data
Base.size(a::HaloArray{T, N, NHALO, SIZE}) where {T, N, NHALO, SIZE} = SIZE
Base.IndexStyle(a::HaloArray) = Base.IndexCartesian()

# use the constructor where `comm` is already in cartesian nprocesses
Base.similar(a::HaloArray{T}) where {T} = HaloArray{T}(comm(a), size(a), nhalo(a))
Base.copy(a::HaloArray) = (b = similar(a); b .= a; b)

Base.@propagate_inbounds @inline function Base.getindex(a::HaloArray{T, N},
                                                     idxs::Vararg{Int, N}) where {T, N}
    _idxs = nhalo(a) .+ idxs
    @boundscheck checkbounds(a.data, _idxs...)
    @inbounds v = a.data[_idxs...]
    return v
end

Base.@propagate_inbounds @inline function Base.setindex!(a::HaloArray{T, N},
                                                   v, idxs::Vararg{Int, N}) where {T, N}
    _idxs = nhalo(a) .+ idxs
    @boundscheck checkbounds(a.data, _idxs...)
    @inbounds a.data[_idxs...] = v
    return v
end

# overload the broadcasting machinery
const HAStyle = Broadcast.ArrayStyle{HaloArray}
Base.BroadcastStyle(::Type{<:HaloArray}) = HAStyle()

# overload materialize! to loop over the active region directly without
# accessing the getindex/setindex! functions. This should ideally have
# the same performance, but I have not checked.
@generated function Base.Broadcast.materialize!(dest::HaloArray{T, N, NHALO, SIZE},
                                                  bc::Broadcast.Broadcasted{<:HAStyle}) where {T, N, NHALO, SIZE}
    quote
        $(Expr(:meta, :inline))
        @inbounds begin
            Base.Cartesian.@nloops $N i dest begin
                (Base.Cartesian.@nref $N dest i) = (Base.Cartesian.@nref $N bc i)
            end
        end
        return dest
    end
end

"""
    source_dest_ranks(a::HaloArray{T, N}, region::NTuple{N, Region}) where {T, N}

Return ranks of neighbouring processors at region `region`.
"""
function source_dest_ranks(a::HaloArray{T, N}, region::NTuple{N, Region}) where {T, N}
    # Determine the direction (dimension) of the shift from the
    # specification of the region. This is the index of the first
    # and unique index of `region` which is either right or left.
    # The displacement argument to Cart_shift depends on whether
    # we consider the LEFT or RIGHT boundaries. When there is no
    # neighbour, MPI returns MPI_PROC_NULL which is typically a
    # negative number.
    direction = findfirst(r -> r in (LEFT, RIGHT), region)
    disp = region[direction] == LEFT ? -1 : 1

    # call the MPI api and return the source and destination ranks
    return MPI.Cart_shift(comm(a), direction - 1, disp)
end

"""
    haloswap!(a::HaloArray{T, N}) where {T, N}

Executes a blocking halo swap between adjacent processes.
"""
function haloswap!(a::HaloArray)
    # we do not need to care about the periodicity and take care of boundary conditions
    # see https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node53.html
    #
    # For every halo region, we need to SEND it to the adjacent processor and
    # RECV from the processor adjacent to the opposite region. This corresponds
    # to a standard shifting of the data across processors and avoids deadlock.
    for (tag, region) in enumerate(a.haloregions)
        source_rank, dest_rank = source_dest_ranks(a, region)
        subsend = a.subarrays[(         region,  SEND)]
        subrecv = a.subarrays[(opposite(region), RECV)]
        MPI.Sendrecv(parent(a),
                     1,
                     subsend,
                     dest_rank,
                     tag,
                     parent(a),
                     1,
                     subrecv,
                     source_rank,
                     tag,
                     comm(a))
    end
    return nothing
end

end