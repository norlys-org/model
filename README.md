## model
>
> Norlys stands for "Ny Oval Representasjon for nordLYS", meaning "New Oval Representation of the northern Lights" in norwegian

`norlys` (pronounced: "noorlus") is a machine learning model using [Spherical Elementary Current Systems](https://link.springer.com/chapter/10.1007/978-3-030-26732-2_2).
Using ground measurements from magnetometers, `norlys` can compute a vector field over the given area reconstructing the magnetic field.

```bash
nix run .#build:wasm -- --dev

# builds wasm in dev mode and runs a web server on 8000
nix run .#serve

# run tests and benchmarks
nix run .#test
nix run .#benchmark
```

### Technologies

Built in [Rust](https://www.rust-lang.org/) and compiles to [WebAssembly](https://webassembly.org/) to be used in [v8](https://v8.dev/) (backend runs on v8 with CloudFlare Workers)
