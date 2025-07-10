use anyhow::{Result, Context};
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{self, Write};
use zip::{ZipArchive, ZipWriter, CompressionMethod};
use chrono::{DateTime, Utc};
use sha2::{Sha256, Digest};
use hex;

use crate::{sbom, signing, tpm};

/// Bundle metadata
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BundleMetadata {
    pub version: String,
    pub created_at: DateTime<Utc>,
    pub git_sha: String,
    pub proof_hash: Option<String>,
    pub sbom_hash: String,
    pub signature_hash: Option<String>,
    pub tpm_attestation: Option<String>,
}

/// Generate a complete compliance bundle
pub async fn generate_bundle(
    output_dir: &Path,
    proof_hash: Option<&str>,
    policy_guard: Option<&Path>,
    sign: bool,
    tpm_attest: bool,
) -> Result<()> {
    println!("Generating LeanEdge-RL compliance bundle...");
    
    // Get git SHA
    let git_sha = get_git_sha()?;
    
    // Generate SBOM
    let sbom_path = output_dir.join("sbom.json");
    sbom::generate_sbom(&sbom_path)?;
    let sbom_hash = calculate_file_hash(&sbom_path)?;
    
    // Build artifacts
    let artifacts = build_artifacts(output_dir).await?;
    
    // Generate bundle filename
    let bundle_name = format!("leanrl_bundle_{}.zip", git_sha);
    let bundle_path = output_dir.join(bundle_name);
    
    // Create ZIP bundle
    create_zip_bundle(&bundle_path, &artifacts, &sbom_path).await?;
    
    // Calculate bundle hash
    let bundle_hash = calculate_file_hash(&bundle_path)?;
    
    // Sign bundle if requested
    let signature_hash = if sign {
        signing::sign_bundle(&bundle_path).await?;
        Some(calculate_file_hash(&bundle_path.with_extension("zip.sig"))?)
    } else {
        None
    };
    
    // Generate TPM attestation if requested
    let tpm_attestation = if tpm_attest {
        Some(tpm::generate_attestation(&bundle_path).await?)
    } else {
        None
    };
    
    // Create metadata
    let metadata = BundleMetadata {
        version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: Utc::now(),
        git_sha,
        proof_hash: proof_hash.map(|s| s.to_string()),
        sbom_hash,
        signature_hash,
        tpm_attestation,
    };
    
    // Write metadata
    let metadata_path = output_dir.join("bundle_metadata.json");
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(&metadata_path, metadata_json)?;
    
    println!("Bundle generated successfully: {}", bundle_path.display());
    println!("Bundle hash: {}", bundle_hash);
    println!("Size: {} bytes", fs::metadata(&bundle_path)?.len());
    
    Ok(())
}

/// Verify bundle integrity
pub async fn verify_bundle(bundle_path: &Path) -> Result<()> {
    println!("Verifying bundle integrity: {}", bundle_path.display());
    
    // Check if bundle exists
    if !bundle_path.exists() {
        anyhow::bail!("Bundle file not found: {}", bundle_path.display());
    }
    
    // Verify ZIP integrity
    verify_zip_integrity(bundle_path)?;
    
    // Extract and verify SBOM
    let temp_dir = tempfile::tempdir()?;
    extract_sbom_from_bundle(bundle_path, &temp_dir.path()).await?;
    
    let sbom_path = temp_dir.path().join("sbom.json");
    if sbom_path.exists() {
        sbom::verify_sbom(&sbom_path)?;
        println!("✓ SBOM verification passed");
    }
    
    // Verify signature if present
    let sig_path = bundle_path.with_extension("zip.sig");
    if sig_path.exists() {
        signing::verify_signature(bundle_path, &sig_path).await?;
        println!("✓ Signature verification passed");
    }
    
    // Verify TPM attestation if present
    let attest_path = bundle_path.with_extension("zip.attest");
    if attest_path.exists() {
        tpm::verify_attestation(&attest_path).await?;
        println!("✓ TPM attestation verification passed");
    }
    
    println!("Bundle verification completed successfully");
    Ok(())
}

/// Build all artifacts for the bundle
async fn build_artifacts(output_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut artifacts = Vec::new();
    
    // Build core library
    println!("Building core library...");
    let core_lib = build_core_library(output_dir).await?;
    artifacts.push(core_lib);
    
    // Build C++ shim
    println!("Building C++ shim...");
    let cshim_lib = build_cshim_library(output_dir).await?;
    artifacts.push(cshim_lib);
    
    // Copy headers
    println!("Copying headers...");
    let headers = copy_headers(output_dir)?;
    artifacts.extend(headers);
    
    // Copy documentation
    println!("Copying documentation...");
    let docs = copy_documentation(output_dir)?;
    artifacts.extend(docs);
    
    Ok(artifacts)
}

/// Build core library
async fn build_core_library(output_dir: &Path) -> Result<PathBuf> {
    // This would typically run `cargo build --release` for the core crate
    // For now, we'll create a placeholder
    let lib_path = output_dir.join("lib").join("libleanrl_core.a");
    fs::create_dir_all(lib_path.parent().unwrap())?;
    
    // Create a placeholder library file
    let mut file = File::create(&lib_path)?;
    file.write_all(b"# Placeholder for libleanrl_core.a")?;
    
    Ok(lib_path)
}

/// Build C++ shim library
async fn build_cshim_library(output_dir: &Path) -> Result<PathBuf> {
    let lib_path = output_dir.join("lib").join("libleanrl_cshim.a");
    fs::create_dir_all(lib_path.parent().unwrap())?;
    
    // Create a placeholder library file
    let mut file = File::create(&lib_path)?;
    file.write_all(b"# Placeholder for libleanrl_cshim.a")?;
    
    Ok(lib_path)
}

/// Copy header files
fn copy_headers(output_dir: &Path) -> Result<Vec<PathBuf>> {
    let headers_dir = output_dir.join("include");
    fs::create_dir_all(&headers_dir)?;
    
    let mut headers = Vec::new();
    
    // Copy C++ header
    let cpp_header = headers_dir.join("leanrl.hpp");
    fs::copy("cshim/include/leanrl.hpp", &cpp_header)?;
    headers.push(cpp_header);
    
    // Copy C header
    let c_header = headers_dir.join("leanrl.h");
    create_c_header(&c_header)?;
    headers.push(c_header);
    
    Ok(headers)
}

/// Copy documentation
fn copy_documentation(output_dir: &Path) -> Result<Vec<PathBuf>> {
    let docs_dir = output_dir.join("docs");
    fs::create_dir_all(&docs_dir)?;
    
    let mut docs = Vec::new();
    
    // Copy README
    let readme = docs_dir.join("README.md");
    fs::copy("README.md", &readme)?;
    docs.push(readme);
    
    // Create API documentation placeholder
    let api_doc = docs_dir.join("api.md");
    let mut file = File::create(&api_doc)?;
    file.write_all(b"# LeanEdge-RL API Documentation\n\n## C API\n\n### lr_init\n### lr_reset\n### lr_step\n### lr_free\n\n## C++ API\n\n### leanrl::Env4x2\n### leanrl::Obs4\n### leanrl::Action2")?;
    docs.push(api_doc);
    
    Ok(docs)
}

/// Create C header file
fn create_c_header(header_path: &Path) -> Result<()> {
    let mut file = File::create(header_path)?;
    
    writeln!(file, "// LeanEdge-RL C API Header")?;
    writeln!(file, "#pragma once")?;
    writeln!(file, "")?;
    writeln!(file, "#include <stdint.h>")?;
    writeln!(file, "#include <stddef.h>")?;
    writeln!(file, "")?;
    writeln!(file, "// Error codes")?;
    writeln!(file, "#define LR_OK             0")?;
    writeln!(file, "#define LR_EBADWEIGHTS   -1")?;
    writeln!(file, "#define LR_EINVSIZE      -2")?;
    writeln!(file, "#define LR_EINVARIANT    -3")?;
    writeln!(file, "#define LR_EOUTOFMEM     -4")?;
    writeln!(file, "")?;
    writeln!(file, "// Opaque environment handle")?;
    writeln!(file, "typedef struct lr_env lr_env_t;")?;
    writeln!(file, "")?;
    writeln!(file, "// C API functions")?;
    writeln!(file, "int lr_init(const uint8_t* weights, size_t len, lr_env_t** out);")?;
    writeln!(file, "int lr_reset(lr_env_t* env, const float* obs, float* action);")?;
    writeln!(file, "int lr_step(lr_env_t* env, const float* obs, float* action);")?;
    writeln!(file, "void lr_free(lr_env_t* env);")?;
    writeln!(file, "int lr_check_invariant(lr_env_t* env, const float* obs, const float* action);")?;
    writeln!(file, "int lr_update_weights(lr_env_t* env, const uint8_t* weights, size_t len);")?;
    writeln!(file, "int lr_get_weights(lr_env_t* env, uint8_t* weights, size_t max_len, size_t* actual_len);")?;
    
    Ok(())
}

/// Create ZIP bundle
async fn create_zip_bundle(
    bundle_path: &Path,
    artifacts: &[PathBuf],
    sbom_path: &Path,
) -> Result<()> {
    let file = File::create(bundle_path)?;
    let mut zip = ZipWriter::new(file);
    
    // Add artifacts
    for artifact in artifacts {
        add_file_to_zip(&mut zip, artifact, "artifacts/").await?;
    }
    
    // Add SBOM
    add_file_to_zip(&mut zip, sbom_path, "").await?;
    
    // Add README
    let readme_content = b"# LeanEdge-RL Compliance Bundle\n\nThis bundle contains:\n- Core library (libleanrl_core.a)\n- C++ shim library (libleanrl_cshim.a)\n- Headers (leanrl.h, leanrl.hpp)\n- Documentation\n- SBOM (sbom.json)\n- Proof metadata\n\nFor integration instructions, see docs/api.md";
    zip.start_file("README.md", CompressionMethod::Deflated)?;
    zip.write_all(readme_content)?;
    
    zip.finish()?;
    Ok(())
}

/// Add file to ZIP
async fn add_file_to_zip(
    zip: &mut ZipWriter<File>,
    file_path: &Path,
    prefix: &str,
) -> Result<()> {
    let file_name = file_path.file_name().unwrap().to_str().unwrap();
    let zip_path = format!("{}{}", prefix, file_name);
    
    zip.start_file(zip_path, CompressionMethod::Deflated)?;
    
    let mut file = File::open(file_path)?;
    io::copy(&mut file, zip)?;
    
    Ok(())
}

/// Verify ZIP integrity
fn verify_zip_integrity(bundle_path: &Path) -> Result<()> {
    let file = File::open(bundle_path)?;
    let mut archive = ZipArchive::new(file)?;
    
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let mut buffer = Vec::new();
        io::copy(&mut file, &mut buffer)?;
        
        // Verify file integrity by checking it can be read
        if buffer.is_empty() && !file.name().ends_with('/') {
            anyhow::bail!("Empty file in bundle: {}", file.name());
        }
    }
    
    Ok(())
}

/// Extract SBOM from bundle
async fn extract_sbom_from_bundle(bundle_path: &Path, output_dir: &Path) -> Result<()> {
    let file = File::open(bundle_path)?;
    let mut archive = ZipArchive::new(file)?;
    
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        if file.name() == "sbom.json" {
            let output_path = output_dir.join("sbom.json");
            let mut output_file = File::create(output_path)?;
            io::copy(&mut file, &mut output_file)?;
            break;
        }
    }
    
    Ok(())
}

/// Get git SHA
fn get_git_sha() -> Result<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .context("Failed to get git SHA")?;
    
    let sha = String::from_utf8(output.stdout)?
        .trim()
        .to_string();
    
    Ok(sha)
}

/// Calculate file hash
fn calculate_file_hash(file_path: &Path) -> Result<String> {
    let mut file = File::open(file_path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    
    let hash = hasher.finalize();
    Ok(hex::encode(hash))
} 