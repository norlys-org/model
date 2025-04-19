{
  description = "norlys model";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/release-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; };
        };

        # bind the actual build script under a distinct name
        buildWasm = pkgs.writeShellApplication {
          name = "build-wasm";
          runtimeInputs = with pkgs; [ rustup wasm-pack ];
          text = ''
            cargo build --target wasm32-unknown-unknown --release
            wasm-pack build --target web
          '';
        };
      in {
        # any app you list here gets run by `nix run .#<key>`
        apps = {
          build-wasm = flake-utils.lib.mkApp { drv = buildWasm; };
        };

        # you can also expose it as a package, if you need to depend on it elsewhere
        packages = {
          inherit buildWasm;
        };

        # drop into a dev shell with your build tool on PATH
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [ rustup wasm-pack ];

          # optional: shell aliases
          shellHook = ''
            echo "Shell ready: run \`build\` to build your WASM."
          '';
        };
      });
}
