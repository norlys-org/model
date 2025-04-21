{
  description = "norlys model";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-24.11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config = { allowUnfree = true; };
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          targets = [ "wasm32-unknown-unknown" ];
        };

        commonBuildInputs = with pkgs; [ rustToolchain wasm-pack ];

        buildWasm = pkgs.writeShellApplication {
          name = "build-wasm";
          runtimeInputs = commonBuildInputs;
          text = ''
            cargo build --target wasm32-unknown-unknown --release
            wasm-pack build --target web
          '';
        };

        test = pkgs.writeShellApplication {
          name = "test";
          runtimeInputs = commonBuildInputs;
          text = ''
            cargo test
          '';
        };

        testWithLogs = pkgs.writeShellApplication {
          name = "test-logs";
          runtimeInputs = commonBuildInputs;
          text = ''
            cargo test -- --nocapture
          '';
        };

        lint = pkgs.writeShellApplication {
          name = "lint";
          runtimeInputs = commonBuildInputs ++ [ pkgs.clippy ];
          text = ''
            cargo clippy -- -D warnings
          '';
        };
      in {
        packages = { inherit buildWasm; };

        apps = {
          "build:wasm" = flake-utils.lib.mkApp { drv = buildWasm; };
          test = flake-utils.lib.mkApp { drv = test; };
          "test:verbose" = flake-utils.lib.mkApp { drv = testWithLogs; };
          lint = flake-utils.lib.mkApp { drv = lint; };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = commonBuildInputs ++ [ pkgs.clippy ];

          shellHook = ''
            echo "norlys model development environment"
          '';
        };
      });
}
