{
  description = "Mini-project-1 polyglot (Rust + Python) development environment";

  inputs = {
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1"; # unstable Nixpkgs
    fenix = {
      url = "https://flakehub.com/f/nix-community/fenix/0.1";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {self, ...} @ inputs: let
    inherit (inputs.nixpkgs) lib;

    supportedSystems = [
      "x86_64-linux"
      "aarch64-linux"
      "aarch64-darwin"
    ];

    forEachSupportedSystem = f:
      lib.genAttrs supportedSystems (
        system:
          f {
            inherit system;
            pkgs = import inputs.nixpkgs {
              inherit system;
              overlays = [self.overlays.default];
            };
          }
      );

    /*
    Change this value ({major}.{min}) to update the Python
    virtual-environment version. When you do this, delete the
    relevant `.venv` directory(ies) and reload the shell.
    */
    version = "3.13";

    concatMajorMinor = v:
      lib.pipe v [
        lib.versions.splitVersion
        (lib.sublist 0 2)
        lib.concatStrings
      ];
  in {
    # ---- Rust toolchain (from your original Rust flake) ----
    overlays.default = final: prev: {
      rustToolchain = with inputs.fenix.packages.${prev.stdenv.hostPlatform.system};
        combine (
          with stable; [
            clippy
            rustc
            cargo
            rustfmt
            rust-src
            rust-analyzer
          ]
        );
    };

    devShells = forEachSupportedSystem (
      {
        pkgs,
        system,
      }: let
        python = pkgs."python${concatMajorMinor version}";
        formatter = self.formatter.${system};

        # ---- shared package lists ----
        rustPackages = with pkgs; [
          rustToolchain
          openssl
          pkg-config
          cargo-deny
          cargo-edit
          cargo-watch
          rust-analyzer
        ];

        pythonPackages = (with python.pkgs; [
          venvShellHook
          pip
          # Project dependencies.  The venv created by venvShellHook does not
          # include system site-packages, so requirements.txt is still the
          # source of truth for the pip-installed versions; these entries make
          # the interpreter-level packages and native libraries available to
          # the shell.
          numpy
          pandas
          scikit-learn
          scipy
          matplotlib
          matplotlib-venn
          pyyaml
          joblib
          pytest
        ]) ++ [
          # numpy's manylinux wheel links against libz; keep it on the shell's
          # LD_LIBRARY_PATH so `import numpy` works inside the venv.
          pkgs.zlib
        ];

        # Report tooling. Typst renders reports/report.typ to a PDF; the
        # generator itself lives in csc869.typst_report.
        docPackages = [pkgs.typst];

        rustEnv = {
          # Required by rust-analyzer
          RUST_SRC_PATH = "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";
        };

        # Python-version mismatch warning (same as your original flake).
        venvVersionWarn = ''
          venvVersionWarn() {
            local venvVersion
            venvVersion="$("$venvDir/bin/python" -c 'import platform; print(platform.python_version())')"

            [[ "$venvVersion" == "${python.version}" ]] && return

            cat <<EOF
          Warning: Python version mismatch: [$venvVersion (venv)] != [${python.version}]
                   Delete '$venvDir' and reload to rebuild for version ${python.version}
          EOF
          }

          venvVersionWarn
        '';

        # ---- shell builders ----

        # Rust-only. Uses mkShell (not NoCC) because rustc needs a linker.
        mkRustShell = {extraPackages ? []}:
          pkgs.mkShell {
            packages = rustPackages ++ [formatter] ++ extraPackages;
            env = rustEnv;
          };

        # Python-only with an auto-created venv. Safe to skip cc here.
        mkPythonShell = {
          venvDir ? ".venv",
          extraPackages ? [],
        }:
          pkgs.mkShellNoCC {
            inherit venvDir;
            postShellHook = venvVersionWarn;
            packages = pythonPackages ++ docPackages ++ [formatter] ++ extraPackages;
          };

        # Both toolchains. mkShell (with cc) so Rust can link.
        mkCombinedShell = {
          venvDir ? ".venv",
          extraPackages ? [],
        }:
          pkgs.mkShell {
            inherit venvDir;
            postShellHook = venvVersionWarn;
            packages = pythonPackages ++ rustPackages ++ docPackages ++ [formatter] ++ extraPackages;
            env = rustEnv;
          };
      in {
        # ---- language-level shells ----
        # `nix develop`               -> both toolchains, venv at ./.venv
        # `nix develop .#rust`        -> Rust only
        # `nix develop .#python`      -> Python only, venv at ./.venv
        default = mkCombinedShell {};
        rust = mkRustShell {};
        python = mkPythonShell {};

        # ---- per-component shells ----
        # Invoke from the repo root so the relative venvDir resolves, e.g.
        #   nix develop .#mp2-rust
        #   nix develop .#mp2-py
        mp2-rust = mkRustShell {};

        mp2-py = mkPythonShell {
          venvDir = "./.venv";
        };

        # Add future components here, e.g.:
        # decision-tree-rs = mkRustShell { };
      }
    );

    formatter = forEachSupportedSystem ({pkgs, ...}: pkgs.nixfmt);
  };
}
