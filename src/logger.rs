use colored::*;
use env_logger::Builder;
use log::{Level, LevelFilter, Record};
use serde::Serialize;
use std::env;
use std::fmt::Formatter;
use std::io::Write;
use csv::Writer;
use std::path::Path;
use std::fs::OpenOptions;


#[derive(Serialize, Debug, Default)]
pub struct ProverPerformanceMetrics {
    // pub num_rows: usize,
    // pub log_rows: u32,
    // pub n: usize,
    // pub params_k: u32,
    pub n: u64, /// Size of the circuit
    pub k: u32,  /// Logaritmic size of the circuit
    pub extended_k: u32, /// size of the extended domain
    pub quotient_poly_degree: usize, 
    pub max_gate_degree: usize, /// the maximum degree of gates in the constraint system
    pub cs_degree: usize, //degree of the constraint system (the maximum degree of all constraints)
    pub num_fixed_columns: usize,
    pub num_advice_columns: usize,
    pub num_instance_columns: usize,
    pub num_selectors: usize,
    pub num_challenges: usize,
    pub minimum_rows: usize, // minimum necessary rows that need to exist in order to account for e.g. blinding factors.
    pub blinding_factors: usize, //number of blinding factors necessary to perfectly blind each of the prover's witness polynomials.
    // pub num_ffts: usize,
    // pub num_msms: usize,
    // pub max_fft_size: usize,
    // pub max_msm_size: usize,
    // pub total_fft_time: f64,
    // pub total_msm_time: f64,
    // pub check_mode: str,
    pub vk_time: f64,
    pub pk_time: f64,
    // pub read_vk_time: f64,
    pub read_pk_time: f64,

    pub setup_time: f64,
    pub proof_time: f64,
    pub verify_time: f64,
}

pub fn write_perf_metrics_to_csv(file_path: &str, metrics: &ProverPerformanceMetrics) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(file_path);

    // Open the file in append mode, create it if it does not exist
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(path)?;

    // Create a CSV writer
    let mut wtr = Writer::from_writer(file);
    
    // Check if the file is empty to determine if we need to write a header
    let file_is_empty = path.metadata()?.len() == 0;

    if file_is_empty {
        // Write the header if the file is empty
        wtr.write_record(&[
            "circuit_size(n)", 
            "log_circuit_size (k)", 
            "extended_k", 
            "quotient_poly_degree", 
            "max_gate_degree",
            "cs_degree", 
            "num_fixed_columns", 
            "num_advice_columns", 
            "num_instance_columns",
            "num_selectors", 
            "num_challenges", 
            "minimum_rows", 
            "blinding_factors",
            "vk_time",
            "pk_time",
            // "read_vk_time",
            "read_pk_time",
            "setup_time", 
            "proof_time", 
            "verify_time"
        ])?;
    }

    // Write the metric record
    wtr.write_record(&[
        metrics.n.to_string(),
        metrics.k.to_string(),
        metrics.extended_k.to_string(),
        metrics.quotient_poly_degree.to_string(),
        metrics.max_gate_degree.to_string(),
        metrics.cs_degree.to_string(),
        metrics.num_fixed_columns.to_string(),
        metrics.num_advice_columns.to_string(),
        metrics.num_instance_columns.to_string(),
        metrics.num_selectors.to_string(),
        metrics.num_challenges.to_string(),
        metrics.minimum_rows.to_string(),
        metrics.blinding_factors.to_string(),
        metrics.vk_time.to_string(),
        metrics.pk_time.to_string(),
        // metrics.read_vk_time.to_string(),
        metrics.read_pk_time.to_string(),

        metrics.setup_time.to_string(),
        metrics.proof_time.to_string(),
        metrics.verify_time.to_string(),
    ])?;

    // Flush the writer to ensure all data is written
    wtr.flush()?;

    println!("Data written to {}", file_path);

    Ok(())
}


#[test]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let metrics = ProverPerformanceMetrics {
    //     n: 10,
    //     k: 2,
    //     extended_k: 3,
    //     quotient_poly_degree: 100,
    //     max_gate_degree: 10,
    //     cs_degree: 20,
    //     num_fixed_columns: 5,
    //     num_advice_columns: 4,
    //     num_instance_columns: 3,
    //     num_selectors: 2,
    //     num_challenges: 1,
    //     minimum_rows: 50,
    //     blinding_factors: 10,
    //     setup_time: 0.5,
    //     proof_time: 1.2,
    //     verify_time: 0.8
    // };
    let metrics: ProverPerformanceMetrics = Default::default();
    write_perf_metrics_to_csv("halo2_circuit.csv", &metrics)?;

    Ok(())
}

/// sets the log level color
#[allow(dead_code)]
pub fn level_color(level: &log::Level, msg: &str) -> String {
    match level {
        Level::Error => msg.red(),
        Level::Warn => msg.yellow(),
        Level::Info => msg.blue(),
        Level::Debug => msg.green(),
        Level::Trace => msg.magenta(),
    }
    .bold()
    .to_string()
}

/// sets the log level text color
pub fn level_text_color(level: &log::Level, msg: &str) -> String {
    match level {
        Level::Error => msg.red(),
        Level::Warn => msg.yellow(),
        Level::Info => msg.white(),
        Level::Debug => msg.white(),
        Level::Trace => msg.white(),
    }
    .bold()
    .to_string()
}

/// sets the log level token
fn level_token(level: &Level) -> &str {
    match *level {
        Level::Error => "E",
        Level::Warn => "W",
        Level::Info => "*",
        Level::Debug => "D",
        Level::Trace => "T",
    }
}

/// sets the log level prefix token
fn prefix_token(level: &Level) -> String {
    format!(
        "{}{}{}",
        "[".blue().bold(),
        level_color(level, level_token(level)),
        "]".blue().bold()
    )
}

/// formats the log
pub fn format(buf: &mut Formatter, record: &Record<'_>) -> Result<(), std::fmt::Error> {
    let sep = format!("\n{} ", " | ".white().bold());
    let level = record.level();
    writeln!(
        buf,
        "{} {}",
        prefix_token(&level),
        level_color(&level, record.args().as_str().unwrap()).replace('\n', &sep),
    )
}

/// initializes the logger
pub fn init_logger() {
    let mut builder = Builder::new();

    builder.format(move |buf, record| {
        writeln!(
            buf,
            "{} [{}, {}] - {}",
            prefix_token(&record.level()),
            //    pretty print UTC time
            chrono::Utc::now()
                .format("%Y-%m-%d %H:%M:%S")
                .to_string()
                .bright_magenta(),
            record.metadata().target(),
            level_text_color(&record.level(), &format!("{}", record.args()))
                .replace('\n', &format!("\n{} ", " | ".white().bold()))
        )
    });
    builder.target(env_logger::Target::Stdout);
    builder.filter(None, LevelFilter::Info);
    if env::var("RUST_LOG").is_ok() {
        builder.parse_filters(&env::var("RUST_LOG").unwrap());
    }
    builder.init();
}
