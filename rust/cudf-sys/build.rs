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

/// Build libcudf from source using cmake.
///
/// Returns `Some((lib_dir, include_dir))` on success, `None` if vendored
/// build should be skipped (env escape hatch or missing source).
#[cfg(feature = "vendored")]
fn build_vendored() -> Option<(PathBuf, PathBuf)> {
    // Escape hatch: CUDF_NO_VENDOR=1 forces system library usage
    if std::env::var("CUDF_NO_VENDOR").unwrap_or_default() == "1" {
        println!("cargo:warning=CUDF_NO_VENDOR=1, skipping vendored build");
        return None;
    }

    // Locate the C++ source directory relative to this crate
    let cpp_dir = PathBuf::from("../../cpp");
    if !cpp_dir.join("CMakeLists.txt").exists() {
        println!(
            "cargo:warning=cpp/CMakeLists.txt not found at {}, falling back to system library",
            cpp_dir.display()
        );
        return None;
    }

    let mut cfg = cmake::Config::new(&cpp_dir);
    cfg.generator("Ninja")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_BENCHMARKS", "OFF")
        .define("CUDF_BUILD_TESTUTIL", "OFF")
        .define("CUDF_BUILD_STREAMS_TEST_UTIL", "OFF");

    // Help cmake find the CUDA toolkit when inside conda/pixi environments
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let cuda_toolkit = PathBuf::from(&prefix);
        if cuda_toolkit.join("bin/nvcc").exists() {
            cfg.define("CMAKE_CUDA_COMPILER", cuda_toolkit.join("bin/nvcc"));
            cfg.define("CUDAToolkit_ROOT", &cuda_toolkit);
        }
    }

    let dst = cfg.build();

    let lib_dir = dst.join("lib");
    let lib64_dir = dst.join("lib64");
    let include_dir = dst.join("include");

    // Emit search paths for both lib/ and lib64/ (cmake may use either)
    if lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    if lib64_dir.exists() {
        println!("cargo:rustc-link-search=native={}", lib64_dir.display());
    }

    // Link libcudf and librmm (rmm-sys skipped these in vendored mode)
    println!("cargo:rustc-link-lib=cudf");
    println!("cargo:rustc-link-lib=rmm");
    println!("cargo:rustc-link-lib=cudart");

    Some((lib_dir, include_dir))
}

fn build_system() -> (Vec<PathBuf>, Vec<PathBuf>) {
    println!("cargo:rustc-link-lib=cudf");
    println!("cargo:rustc-link-lib=cudart");

    let include_paths = discover_include_paths("CUDF_ROOT");
    let lib_paths = discover_lib_paths("CUDF_ROOT");

    for p in &lib_paths {
        println!("cargo:rustc-link-search=native={}", p.display());
    }

    (include_paths, lib_paths)
}

fn main() {
    // Determine include paths based on build mode
    let include_paths: Vec<PathBuf>;

    #[cfg(feature = "vendored")]
    {
        if let Some((_lib_dir, vendored_include)) = build_vendored() {
            include_paths = vec![vendored_include.clone(), vendored_include.join("rapids")];
        } else {
            let (sys_includes, _) = build_system();
            include_paths = sys_includes;
        }
    }

    #[cfg(not(feature = "vendored"))]
    {
        let (sys_includes, _) = build_system();
        include_paths = sys_includes;
    }

    let mut build = cxx_build::bridges([
        "src/lib.rs",
        "src/ast.rs",
        "src/binaryop.rs",
        "src/compaction.rs",
        "src/concatenate.rs",
        "src/contiguous_split.rs",
        "src/copying.rs",
        "src/datetime.rs",
        "src/dictionary.rs",
        "src/filling.rs",
        "src/groupby.rs",
        "src/hashing.rs",
        "src/io.rs",
        "src/join.rs",
        "src/labeling.rs",
        "src/lists.rs",
        "src/merge.rs",
        "src/partitioning.rs",
        "src/quantile.rs",
        "src/reduction.rs",
        "src/replace.rs",
        "src/reshape.rs",
        "src/rolling.rs",
        "src/search.rs",
        "src/sorting.rs",
        "src/strings.rs",
        "src/transform.rs",
        "src/unary.rs",
    ]);
    build.include("include");
    for p in &include_paths {
        build.include(p);
    }
    build
        .file("cpp/lib.cpp")
        .file("cpp/ast.cpp")
        .file("cpp/binaryop.cpp")
        .file("cpp/compaction.cpp")
        .file("cpp/concatenate.cpp")
        .file("cpp/contiguous_split.cpp")
        .file("cpp/copying.cpp")
        .file("cpp/datetime.cpp")
        .file("cpp/dictionary.cpp")
        .file("cpp/filling.cpp")
        .file("cpp/groupby.cpp")
        .file("cpp/hashing.cpp")
        .file("cpp/io.cpp")
        .file("cpp/join.cpp")
        .file("cpp/labeling.cpp")
        .file("cpp/lists.cpp")
        .file("cpp/merge.cpp")
        .file("cpp/partitioning.cpp")
        .file("cpp/quantile.cpp")
        .file("cpp/reduction.cpp")
        .file("cpp/replace.cpp")
        .file("cpp/reshape.cpp")
        .file("cpp/rolling.cpp")
        .file("cpp/search.cpp")
        .file("cpp/sorting.cpp")
        .file("cpp/strings.cpp")
        .file("cpp/transform.cpp")
        .file("cpp/unary.cpp")
        .flag("-std=c++20")
        .flag("-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
        .compile("cudf-sys");

    println!("cargo:rerun-if-changed=include/");
    println!("cargo:rerun-if-changed=cpp/");
    println!("cargo:rerun-if-changed=src/");
}
