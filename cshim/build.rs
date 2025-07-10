fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("src/cxxbridge.cc")
        .flag_if_supported("-std=c++17")
        .compile("leanrl_cshim");
    
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/cxxbridge.cc");
    println!("cargo:rerun-if-changed=include/leanrl.hpp");
} 