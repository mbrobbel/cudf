// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

/// Discover include paths for a given library.
///
/// Tries in order:
/// 1. `{LIB}_ROOT` env var (e.g. `RMM_ROOT`)
/// 2. `CONDA_PREFIX` env var
fn discover_include_paths(root_env: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(root) = std::env::var(root_env) {
        let root = PathBuf::from(root);
        paths.push(root.join("include"));
        // rapids-cmake installs headers under include/rapids/ in some configurations
        paths.push(root.join("include/rapids"));
    }

    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let prefix = PathBuf::from(prefix);
        paths.push(prefix.join("include"));
        paths.push(prefix.join("include/rapids"));

        // Architecture-specific paths
        let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        let target_dir = format!("targets/{arch}-linux/include");
        paths.push(prefix.join(target_dir));
    }

    paths
}

/// Discover library search paths.
fn discover_lib_paths(root_env: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(root) = std::env::var(root_env) {
        let root = PathBuf::from(root);
        paths.push(root.join("lib"));
        paths.push(root.join("lib64"));
    }

    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let prefix = PathBuf::from(prefix);
        paths.push(prefix.join("lib"));
        paths.push(prefix.join("lib64"));

        let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        let target_dir = format!("targets/{arch}-linux/lib");
        paths.push(prefix.join(target_dir));
    }

    paths
}

fn main() {
    let vendored = cfg!(feature = "vendored");

    // In vendored mode, cudf-sys builds libcudf from source (which includes
    // librmm via rapids_cpm_rmm). cudf-sys will emit the link directives for
    // both libcudf and librmm. We skip linking here to avoid duplicate symbols.
    if !vendored {
        println!("cargo:rustc-link-lib=rmm");
        println!("cargo:rustc-link-lib=cudart");

        let lib_paths = discover_lib_paths("RMM_ROOT");
        for p in &lib_paths {
            println!("cargo:rustc-link-search=native={}", p.display());
        }

        // Also check CUDA_ROOT / CUDA_PATH for cudart
        for env in &["CUDA_ROOT", "CUDA_PATH", "CUDA_HOME"] {
            if let Ok(root) = std::env::var(env) {
                let root = PathBuf::from(root);
                println!(
                    "cargo:rustc-link-search=native={}",
                    root.join("lib64").display()
                );
                println!(
                    "cargo:rustc-link-search=native={}",
                    root.join("lib").display()
                );
            }
        }
    }

    let include_paths = discover_include_paths("RMM_ROOT");

    let mut build = cxx_build::bridge("src/lib.rs");
    build.include("include");
    for p in &include_paths {
        build.include(p);
    }
    build
        .flag("-std=c++20")
        .flag("-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
        .file("cpp/lib.cpp")
        .compile("rmm-sys");

    println!("cargo:rerun-if-changed=include/");
    println!("cargo:rerun-if-changed=cpp/");
    println!("cargo:rerun-if-changed=src/lib.rs");
}
