import HaloArrays: swapregions,
                   LEFT, RIGHT, ALL, CENTER,
                   subarray_size,
                   subarray_start,
                   SEND, RECV

using Test

@testset "swapregions                            " begin
    # 1d
    topology    = (  10,)
    isperiodic  = (true,)
    @test swapregions(topology, isperiodic, true) == [(LEFT,), (RIGHT,)]

    topology    = (   10,)
    isperiodic  = (false,)
    @test swapregions(topology, isperiodic, true) == [(LEFT,), (RIGHT,)]

    topology    = (   1,)
    isperiodic  = (true,)
    @test swapregions(topology, isperiodic, true) == [(LEFT,), (RIGHT,)]

    topology    = (    1,)
    isperiodic  = (false,)
    @test swapregions(topology, isperiodic, true) == []

    # 2d
    topology    = (   1,     1)
    isperiodic  = (true, false)
    @test swapregions(topology, isperiodic, true) == [(LEFT,   CENTER),
                                                      (RIGHT,  CENTER)]

    topology    = (   10,   10)
    isperiodic  = (false, false)
    @test swapregions(topology, isperiodic, true) == [(LEFT,   CENTER),
                                                      (RIGHT,  CENTER),
                                                      (CENTER, LEFT),
                                                      (CENTER, RIGHT)]

    topology    = (  2,     2)
    isperiodic  = (true, true)
    @test swapregions(topology, isperiodic, true) == [(LEFT,   CENTER),
                                                      (RIGHT,  CENTER),
                                                      (CENTER, LEFT),
                                                      (CENTER, RIGHT)]

    topology    = (  1,     1)
    isperiodic  = (true, true)
    @test swapregions(topology, isperiodic, true) == [(LEFT,   CENTER),
                                                      (RIGHT,  CENTER),
                                                      (CENTER, LEFT),
                                                      (CENTER, RIGHT)]

    topology    = (  1,      1)
    isperiodic  = (false, true)
    @test swapregions(topology, isperiodic, true) == [(CENTER, LEFT),
                                                      (CENTER, RIGHT)]

    # 3d
    topology    = (    1,     1,     1)
    isperiodic  = (false, false, false)
    @test swapregions(topology, isperiodic, true) == []

    topology    = (  1,      10,     1)
    isperiodic  = (false, false, false)
    @test swapregions(topology, isperiodic, true) == [(CENTER,  LEFT, CENTER),
                                                      (CENTER, RIGHT, CENTER)]

    topology    = (  1,      10,    10)
    isperiodic  = (false, false, false)
    @test swapregions(topology, isperiodic, true) == [(CENTER,   LEFT, CENTER),
                                                      (CENTER,  RIGHT, CENTER),
                                                      (CENTER, CENTER, LEFT),
                                                      (CENTER, CENTER, RIGHT)]

    topology    = (   1,    1,     10)
    isperiodic  = (true, false, false)
    @test swapregions(topology, isperiodic, true) == [(  LEFT, CENTER, CENTER),
                                                      ( RIGHT, CENTER, CENTER),
                                                      (CENTER, CENTER, LEFT),
                                                      (CENTER, CENTER, RIGHT)]

    topology    = (   1,    10,    1)
    isperiodic  = (true, false, true)
    @test swapregions(topology, isperiodic, true) == [(LEFT,   CENTER, CENTER),
                                                      (RIGHT,  CENTER, CENTER),
                                                      (CENTER, LEFT,   CENTER),
                                                      (CENTER, RIGHT,  CENTER),
                                                      (CENTER, CENTER, LEFT),
                                                      (CENTER, CENTER, RIGHT)]

    topology    = (   1,    10,    1)
    isperiodic  = (true, false, true)
    @test swapregions(topology, isperiodic, false) == [(LEFT,  ALL,   ALL),
                                                       (RIGHT, ALL,   ALL),
                                                       (ALL,   LEFT,  ALL),
                                                       (ALL,   RIGHT, ALL),
                                                       (ALL,   ALL,   LEFT),
                                                       (ALL,   ALL,   RIGHT)]
end

@testset "subarray size/start                    " begin
    # 1d

    # 1 2 3 4 5 6 7 8 9
    # h h 1 2 3 4 5 h h
    @test subarray_size((LEFT,),   (5,), (2,)) == (2,)
    @test subarray_size((RIGHT,),  (5,), (2,)) == (2,)
    @test subarray_size((CENTER,), (5,), (2,)) == (5,)
    @test subarray_size((ALL,),    (5,), (2,)) == (9,)

    # 1 2 3 4 5 6 7 8 9 10 11
    # h h h 1 2 3 4 5 h  h  h
    @test subarray_start((LEFT,),   SEND, (5,), (3,)) == (4,)
    @test subarray_start((RIGHT,),  SEND, (5,), (3,)) == (6,)
    @test subarray_start((CENTER,), SEND, (5,), (3,)) == (4,)
    @test subarray_start((ALL,),    SEND, (5,), (3,)) == (1,)
    @test subarray_start((LEFT,),   RECV, (5,), (3,)) == (1,)
    @test subarray_start((RIGHT,),  RECV, (5,), (3,)) == (9,)
    @test subarray_start((CENTER,), RECV, (5,), (3,)) == (4,)
    @test subarray_start((ALL,),    RECV, (5,), (3,)) == (1,)

    # 2d
    @test subarray_size((LEFT,   ALL),    (5, 4), (2, 2)) == (2, 8)
    @test subarray_size((LEFT,   CENTER), (5, 4), (2, 2)) == (2, 4)
    @test subarray_size((CENTER, LEFT),   (5, 4), (2, 1)) == (5, 1)
    @test subarray_size((ALL,    RIGHT),  (5, 4), (2, 1)) == (9, 1)

    # 3d
    @test subarray_size((LEFT,   ALL,    ALL),    (5, 4, 3), (1, 2, 3)) == (1, 8, 9)
    @test subarray_size((LEFT,   CENTER, CENTER), (5, 4, 3), (1, 2, 3)) == (1, 4, 3)
    @test subarray_size((CENTER, LEFT,   ALL),    (5, 4, 3), (1, 2, 3)) == (5, 2, 9)
    @test subarray_size((CENTER, LEFT,   CENTER), (5, 4, 3), (1, 2, 3)) == (5, 2, 3)
    @test subarray_size((CENTER, CENTER, RIGHT),  (5, 4, 3), (1, 2, 3)) == (5, 4, 3)

    @test subarray_start((LEFT,   ALL,    ALL),    SEND, (5, 4, 3), (1, 2, 3)) == (2, 1, 1)
    @test subarray_start((LEFT,   ALL,    ALL),    RECV, (5, 4, 3), (1, 2, 3)) == (1, 1, 1)
    @test subarray_start((RIGHT,  ALL,    ALL),    SEND, (5, 4, 3), (1, 2, 3)) == (6, 1, 1)
    @test subarray_start((RIGHT,  ALL,    ALL),    RECV, (5, 4, 3), (1, 2, 3)) == (7, 1, 1)
    @test subarray_start((CENTER, LEFT,   ALL),    SEND, (5, 4, 3), (1, 2, 3)) == (2, 3, 1)
    @test subarray_start((CENTER, RIGHT,  ALL),    RECV, (5, 4, 3), (1, 2, 3)) == (2, 7, 1)
    @test subarray_start((CENTER, RIGHT,  CENTER), RECV, (5, 4, 3), (1, 2, 3)) == (2, 7, 4)
    # these should be enough. the algorithm works for any number of dimensions
end