push!(LOAD_PATH,"src/")
push!(LOAD_PATH,"../src/")

using Documenter
using MultivariateChebyshev

makedocs(
    sitename = "MultivariateChebyshev",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [MultivariateChebyshev],
    pages = [
        "index.md",
        "pages/matrix_methods.md",
        "pages/chebyshev.md",
        "pages/function_index.md",
    ],
)

deploydocs(
    repo = "https://github.com/sschlenkrich/MultivariateChebyshev.jl",
)
