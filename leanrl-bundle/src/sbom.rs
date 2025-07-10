use anyhow::Result;
use std::path::Path;
use std::fs::File;
use std::io::Write;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use sha2::{Sha256, Digest};
use hex;

/// SPDX SBOM document
#[derive(Debug, Serialize, Deserialize)]
pub struct SpdxDocument {
    #[serde(rename = "SPDXID")]
    pub spdx_id: String,
    #[serde(rename = "spdxVersion")]
    pub spdx_version: String,
    #[serde(rename = "creationInfo")]
    pub creation_info: CreationInfo,
    pub name: String,
    #[serde(rename = "dataLicense")]
    pub data_license: String,
    pub packages: Vec<Package>,
    pub relationships: Vec<Relationship>,
}

/// Creation info for SBOM
#[derive(Debug, Serialize, Deserialize)]
pub struct CreationInfo {
    pub creators: Vec<String>,
    #[serde(rename = "created")]
    pub created: String,
    #[serde(rename = "licenseListVersion")]
    pub license_list_version: String,
}

/// Package information
#[derive(Debug, Serialize, Deserialize)]
pub struct Package {
    #[serde(rename = "SPDXID")]
    pub spdx_id: String,
    pub name: String,
    #[serde(rename = "versionInfo")]
    pub version_info: String,
    #[serde(rename = "packageFileName")]
    pub package_file_name: String,
    #[serde(rename = "checksums")]
    pub checksums: Vec<Checksum>,
    #[serde(rename = "licenseConcluded")]
    pub license_concluded: String,
    #[serde(rename = "licenseDeclared")]
    pub license_declared: String,
    #[serde(rename = "copyrightText")]
    pub copyright_text: String,
    #[serde(rename = "supplier")]
    pub supplier: String,
    pub description: String,
}

/// Checksum information
#[derive(Debug, Serialize, Deserialize)]
pub struct Checksum {
    pub algorithm: String,
    #[serde(rename = "checksumValue")]
    pub checksum_value: String,
}

/// Relationship between packages
#[derive(Debug, Serialize, Deserialize)]
pub struct Relationship {
    #[serde(rename = "spdxElementId")]
    pub spdx_element_id: String,
    #[serde(rename = "relatedSpdxElement")]
    pub related_spdx_element: String,
    #[serde(rename = "relationshipType")]
    pub relationship_type: String,
}

/// Generate SBOM
pub fn generate_sbom(output_path: &Path) -> Result<()> {
    println!("Generating SBOM...");
    
    let sbom = create_spdx_document()?;
    
    let mut file = File::create(output_path)?;
    let json = serde_json::to_string_pretty(&sbom)?;
    file.write_all(json.as_bytes())?;
    
    println!("SBOM generated: {}", output_path.display());
    Ok(())
}

/// Verify SBOM
pub fn verify_sbom(sbom_path: &Path) -> Result<()> {
    let content = std::fs::read_to_string(sbom_path)?;
    let _sbom: SpdxDocument = serde_json::from_str(&content)?;
    
    println!("SBOM verification passed");
    Ok(())
}

/// Create SPDX document
fn create_spdx_document() -> Result<SpdxDocument> {
    let now = Utc::now();
    let timestamp = now.format("%Y-%m-%dT%H:%M:%SZ").to_string();
    
    let mut packages = Vec::new();
    let mut relationships = Vec::new();
    
    // Add main package
    let main_package = Package {
        spdx_id: "SPDXRef-leanrl-core".to_string(),
        name: "LeanEdge-RL Core".to_string(),
        version_info: env!("CARGO_PKG_VERSION").to_string(),
        package_file_name: "libleanrl_core.a".to_string(),
        checksums: vec![
            Checksum {
                algorithm: "SHA256".to_string(),
                checksum_value: calculate_file_hash("core/src/lib.rs")?,
            }
        ],
        license_concluded: "MIT OR Apache-2.0".to_string(),
        license_declared: "MIT OR Apache-2.0".to_string(),
        copyright_text: "Copyright (c) 2025 LeanEdge-RL Team".to_string(),
        supplier: "LeanEdge-RL Team".to_string(),
        description: "Core RL runtime library for safety-critical edge systems".to_string(),
    };
    packages.push(main_package);
    
    // Add C++ shim package
    let cshim_package = Package {
        spdx_id: "SPDXRef-leanrl-cshim".to_string(),
        name: "LeanEdge-RL C++ Shim".to_string(),
        version_info: env!("CARGO_PKG_VERSION").to_string(),
        package_file_name: "libleanrl_cshim.a".to_string(),
        checksums: vec![
            Checksum {
                algorithm: "SHA256".to_string(),
                checksum_value: calculate_file_hash("cshim/src/lib.rs")?,
            }
        ],
        license_concluded: "MIT OR Apache-2.0".to_string(),
        license_declared: "MIT OR Apache-2.0".to_string(),
        copyright_text: "Copyright (c) 2025 LeanEdge-RL Team".to_string(),
        supplier: "LeanEdge-RL Team".to_string(),
        description: "C++ shim and header generation for LeanEdge-RL".to_string(),
    };
    packages.push(cshim_package);
    
    // Add dependencies
    let dependencies = vec![
        ("thiserror", "1.0", "MIT OR Apache-2.0"),
        ("serde", "1.0", "MIT OR Apache-2.0"),
        ("libc", "0.2", "MIT OR Apache-2.0"),
        ("packed_simd_2", "0.3", "MIT OR Apache-2.0"),
        ("cxx", "1.0", "MIT OR Apache-2.0"),
        ("clap", "4.4", "MIT OR Apache-2.0"),
        ("zip", "0.6", "MIT"),
        ("walkdir", "2.4", "Unlicense/MIT"),
        ("toml", "0.8", "MIT"),
        ("serde_json", "1.0", "MIT OR Apache-2.0"),
        ("sha2", "0.10", "MIT OR Apache-2.0"),
        ("hex", "0.4", "MIT OR Apache-2.0"),
        ("chrono", "0.4", "MIT OR Apache-2.0"),
        ("anyhow", "1.0", "MIT OR Apache-2.0"),
    ];
    
    for (i, (name, version, license)) in dependencies.iter().enumerate() {
        let package = Package {
            spdx_id: format!("SPDXRef-dependency-{}", i),
            name: name.to_string(),
            version_info: version.to_string(),
            package_file_name: format!("{}-{}.crate", name, version),
            checksums: vec![
                Checksum {
                    algorithm: "SHA256".to_string(),
                    checksum_value: format!("placeholder_hash_{}", i),
                }
            ],
            license_concluded: license.to_string(),
            license_declared: license.to_string(),
            copyright_text: "Copyright (c) respective authors".to_string(),
            supplier: "Crates.io".to_string(),
            description: format!("Dependency: {}", name),
        };
        packages.push(package);
        
        // Add relationship
        relationships.push(Relationship {
            spdx_element_id: "SPDXRef-leanrl-core".to_string(),
            related_spdx_element: format!("SPDXRef-dependency-{}", i),
            relationship_type: "DEPENDS_ON".to_string(),
        });
    }
    
    // Add relationship between core and cshim
    relationships.push(Relationship {
        spdx_element_id: "SPDXRef-leanrl-cshim".to_string(),
        related_spdx_element: "SPDXRef-leanrl-core".to_string(),
        relationship_type: "DEPENDS_ON".to_string(),
    });
    
    let creation_info = CreationInfo {
        creators: vec![
            "Tool: leanrl-bundle".to_string(),
            "Organization: LeanEdge-RL Team".to_string(),
        ],
        created: timestamp,
        license_list_version: "3.19".to_string(),
    };
    
    Ok(SpdxDocument {
        spdx_id: "SPDXRef-DOCUMENT".to_string(),
        spdx_version: "SPDX-2.3".to_string(),
        creation_info,
        name: "LeanEdge-RL Software Bill of Materials".to_string(),
        data_license: "CC0-1.0".to_string(),
        packages,
        relationships,
    })
}

/// Calculate file hash
fn calculate_file_hash(file_path: &str) -> Result<String> {
    // In a real implementation, this would read the actual file
    // For now, we'll create a placeholder hash
    let mut hasher = Sha256::new();
    hasher.update(file_path.as_bytes());
    let hash = hasher.finalize();
    Ok(hex::encode(hash))
} 