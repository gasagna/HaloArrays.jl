import HaloArrays: HaloArray

using BenchmarkTools
using Test
using MPI

MPI.Init()

@testset "broadcasting speed                     " begin
    a = HaloArray{Float64}(MPI.COMM_WORLD,
                           (    1,     1,     1),
                           (false, false, false),
                           (  100,   100,   100),
                           (    1,     1,     1))
    b = copy(a)
    c = copy(a)
    d = copy(a)
    e = copy(a)

    # test broadcasting is as fast as looping the underlying parent array
    foo(a, b, c, d, e) = (a .= 2.0.*b .+ 3.0.*c .+ 4.0.*d .+ 5.0.*e; a)

    function bar(a, b, c, d, e)
        _a = parent(a)
        _b = parent(b)
        _c = parent(c)
        _d = parent(d)
        _e = parent(e)
        @inbounds for k = 2:101
            for j = 2:101
                @simd for i = 2:101
                    _a[i, j, k] = 2.0*_b[i, j, k] + 3.0.*_c[i, j, k] + 4.0.*_d[i, j, k] + 5.0.*_e[i, j, k]
                end
            end
        end
        return a
    end

    t_broadcast = @belapsed $foo($a, $b, $c, $d, $e)
    t_looping   = @belapsed $bar($a, $b, $c, $d, $e)
    
    # two percent difference should not trigger random fails
    @test abs(t_broadcast - t_looping)/t_looping < 0.02

    # test broadcasting is as fast as looping the underlying parent array
    foo2(a, b) = (a .= b; a)

    function bar2(a, b)
        _a = parent(a)
        _b = parent(b)
        @inbounds for k = 2:101
            for j = 2:101
                @simd for i = 2:101
                    _a[i, j, k] = _b[i, j, k]
                end
            end
        end
        return a
    end

    t_broadcast = @belapsed $foo2($a, $b)
    t_looping   = @belapsed $bar2($a, $b)
    
    # two percent difference should not trigger random fails
    @test abs(t_broadcast - t_looping)/t_looping < 0.02

end

MPI.Finalize()