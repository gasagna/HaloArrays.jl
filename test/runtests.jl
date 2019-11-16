# list of filenames and number of processors 
# use `0` for serial code
tests = [ ("test_utils.jl",     0),
          ("test_haloarray.jl", 2),
          ("test_swap.jl",      4)]

for (filename, nprocs) in tests
    if nprocs == 0
        run(`julia $filename`)
        Base.with_output_color(:green, stdout) do io
            println(io, "\tSUCCESS: $filename - with $nprocs processors")
        end
    else
        try
            run(`mpirun -np $nprocs julia $filename`)
            Base.with_output_color(:green, stdout) do io
                println(io, "\tSUCCESS: $filename - with $nprocs processors")
            end
        catch ex
            Base.with_output_color(:red, stderr) do io
                println(io, "\tERROR: $filename - with $nprocs processors")
                showerror(io, ex, backtrace())
            end
        end
    end
end