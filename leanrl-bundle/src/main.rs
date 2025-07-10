use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::Result;

mod bundle;
mod sbom;
mod signing;
mod tpm;

#[derive(Parser)]
#[command(name = "leanrl-bundle")]
#[command(about = "Generate compliance bundles for LeanEdge-RL")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,
    
    #[arg(long)]
    sign: bool,
    
    #[arg(long)]
    tpm_attest: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a complete compliance bundle
    Generate {
        #[arg(short, long)]
        proof_hash: Option<String>,
        
        #[arg(short, long)]
        policy_guard: Option<PathBuf>,
    },
    
    /// Generate SBOM only
    Sbom {
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    
    /// Verify bundle integrity
    Verify {
        #[arg(short, long)]
        bundle: PathBuf,
    },
    
    /// Sign bundle with Sigstore
    Sign {
        #[arg(short, long)]
        bundle: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Generate { proof_hash, policy_guard } => {
            bundle::generate_bundle(
                &cli.output_dir,
                proof_hash.as_deref(),
                policy_guard.as_deref(),
                cli.sign,
                cli.tpm_attest,
            ).await?;
        }
        
        Commands::Sbom { output } => {
            let output_path = output.unwrap_or_else(|| PathBuf::from("sbom.json"));
            sbom::generate_sbom(&output_path)?;
        }
        
        Commands::Verify { bundle } => {
            bundle::verify_bundle(&bundle).await?;
        }
        
        Commands::Sign { bundle } => {
            signing::sign_bundle(&bundle).await?;
        }
    }
    
    Ok(())
} 