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

        manifest = (pkgs.lib.importTOML ./Cargo.toml).package;
      in {
        defaultPackage = naersk'.buildPackage {
          name = manifest.name;
          version = manifest.version;

          src = ./.;

          buildInputs = with pkgs; [ wasm-pack ];
          CARGO_BUILD_TARGET = "wasm32-unknown-unknown";

          buildPhase = ''
            export CARGO_TARGET_DIR=$TMPDIR/cargo-target

            # compile and generate JS bindings directly into $out
            wasm-pack build \
              --target web \
              --release \
              --out-dir $out;
          '';
        };

        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [ rustc cargo clippy ];
          shellHook = ''
            echo "norlys model development environment"
          '';
        };
      });
}
