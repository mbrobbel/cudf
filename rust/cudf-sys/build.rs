// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

/// Discover include paths for a given library.
///
/// Tries in order:
/// 1. `{LIB}_ROOT` env var (e.g. `CUDF_ROOT`)
/// 2. `CONDA_PREFIX` env var
fn discover_include_paths(root_env: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(root) = std::env::var(root_env) {
        let root = PathBuf::from(root);
        paths.push(root.join("include"));
        paths.push(root.join("include/rapids"));
    }

    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let prefix = PathBuf::from(prefix);
        paths.push(prefix.join("include"));
        paths.push(prefix.join("include/rapids"));

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
    println!("cargo:rustc-link-lib=cudf");

    let include_paths = discover_include_paths("CUDF_ROOT");
    let lib_paths = discover_lib_paths("CUDF_ROOT");

    for p in &lib_paths {
        println!("cargo:rustc-link-search=native={}", p.display());
    }

    let mut build = cxx_build::bridge("src/lib.rs");
    build.include("include");
    for p in &include_paths {
        build.include(p);
    }
    build
        .file("cpp/lib.cpp")
        .flag("-std=c++20")
        .flag("-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
        .compile("cudf-sys");

    println!("cargo:rerun-if-changed=include/");
    println!("cargo:rerun-if-changed=cpp/");
    println!("cargo:rerun-if-changed=src/");
}
