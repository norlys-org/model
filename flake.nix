{
  description = "norlys model";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-24.11";
    flake-utils.url = "github:numtide/flake-utils";
    naersk.url = "github:nix-community/naersk";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, flake-utils, rust-overlay, naersk, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config = { allowUnfree = true; };
        };

        toolchain = pkgs.rust-bin.stable.latest.default.override {
          targets = [ "wasm32-unknown-unknown" ];
        };
        naersk' = pkgs.callPackage naersk {
          cargo = toolchain;
          rustc = toolchain;
        };

        buildWasmDrv = pkgs.writeShellApplication {
          name = "build-wasm";
          runtimeInputs = [ pkgs.wasm-pack toolchain ];
          text = ''
            echo "\$ wasm-pack build --target web $*"
            wasm-pack build --target web "$@";
          '';
        };

        manifest = (pkgs.lib.importTOML ./Cargo.toml).package;
      in {
        defaultPackage = naersk'.buildPackage {
          name = manifest.name;
          version = manifest.version;

          src = ./.;

          CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
        };

        apps = {
          "build:wasm" = flake-utils.lib.mkApp { drv = buildWasmDrv; };

          test = flake-utils.lib.mkApp {
            drv = pkgs.writeShellApplication {
              name = "test";
              runtimeInputs = [ toolchain ];
              text = ''
                echo "$ cargo test"
                cargo test -- --nocapture
              '';
            };
          };

          benchmark = flake-utils.lib.mkApp {
            drv = pkgs.writeShellApplication {
              name = "benchmark";
              runtimeInputs = [ toolchain ];
              text = ''
                echo "$ cargo bench"
                cargo bench
              '';
            };
          };

          serve = flake-utils.lib.mkApp {
            drv = pkgs.writeShellApplication {
              name = "serve";
              runtimeInputs = [ pkgs.python3 ];
              text = ''
                ${buildWasmDrv}/bin/build-wasm --dev
                echo "$ python -m http.server"
                python -m http.server
              '';
            };
          };
        };

        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [ rustc cargo clippy ];
          shellHook = ''
            echo "norlys model development environment"
          '';
        };
      });
}
