//! Fused KZG setup/prove path for zkInfer. Ported from ccgrid to upstream.
use crate::circuit::CheckMode;
use crate::graph::{GraphCircuit, GraphWitness};
use crate::pfsys::{create_proof_circuit, save_pk, save_vk, ProofSplitCommit};
use halo2_proofs::plonk::{keygen_pk, keygen_vk_custom, Circuit};
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy as KZGSingleStrategy;
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "python-bindings")]
use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;

const STAGE_TRACE_VERSION: u32 = 2;
const STAGE_TRACE_FILENAME: &str = "fused_stage_transitions.csv";

/// Timing information returned by the fused cold path.
#[derive(Debug)]
pub(crate) struct FusedSetupAndProveResult {
    /// Includes loading prover parameters/SRS and generating VK + PK.
    pub keygen_time_s: f64,
    /// Time spent creating the proof, excluding proof serialization.
    pub prove_time_s: f64,
    pub proof_persist_time_s: f64,
    pub key_persist_time_s: f64,
    pub total_time_s: f64,
}

fn stage_trace_path(status_path: &Option<PathBuf>) -> Option<PathBuf> {
    status_path
        .as_ref()
        .map(|path| path.with_file_name(STAGE_TRACE_FILENAME))
}

fn initialize_stage_trace(status_path: &Option<PathBuf>) -> Result<(), Box<dyn Error>> {
    let Some(path) = stage_trace_path(status_path) else {
        return Ok(());
    };

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut file = fs::File::create(path)?;
    writeln!(file, "timestamp_unix_us,elapsed_s,stage,event")?;
    file.flush()?;
    Ok(())
}

fn append_stage_trace(
    status_path: &Option<PathBuf>,
    stage_trace_elapsed_base_s: f64,
    total_start: &Instant,
    stage: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(path) = stage_trace_path(status_path) else {
        return Ok(());
    };

    let timestamp_unix_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_micros();
    let elapsed_s = stage_trace_elapsed_base_s + total_start.elapsed().as_secs_f64();

    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{timestamp_unix_us},{elapsed_s:.9},{stage},enter")?;
    file.flush()?;
    Ok(())
}

fn publish_stage(
    status_path: &Option<PathBuf>,
    stage_trace_elapsed_base_s: f64,
    total_start: &Instant,
    stage: &str,
) -> Result<(), Box<dyn Error>> {
    if let Some(path) = status_path {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, stage)?;
    }

    append_stage_trace(
        status_path,
        stage_trace_elapsed_base_s,
        total_start,
        stage,
    )?;
    Ok(())
}

fn write_keygen_times(vk_time_s: f64, pk_time_s: f64) -> Result<(), Box<dyn Error>> {
    let mut file = fs::File::create("keygen_times.txt")?;
    writeln!(file, "{vk_time_s},{pk_time_s}")?;
    file.flush()?;
    Ok(())
}

/// Generate setup artifacts and immediately prove with the in-memory proving
/// key. This is intentionally a cold-path primitive: callers should use the
/// normal `prove` entrypoint when a cached proving key already exists.
///
/// The existing file-oriented flow performs setup, serializes the PK, returns
/// to Python, then `prove` reads/deserializes the same PK and reloads the SRS.
/// This function keeps the circuit, prover parameters, and PK alive across the
/// setup -> prove boundary. It still persists PK/VK after proving so they can
/// be reused by future warm requests.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_and_prove(
    witness_path: PathBuf,
    compiled_circuit_path: PathBuf,
    vk_path: PathBuf,
    pk_path: PathBuf,
    proof_path: Option<PathBuf>,
    srs_path: Option<PathBuf>,
    check_mode: CheckMode,
    disable_selector_compression: bool,
    status_path: Option<PathBuf>,
    stage_trace_elapsed_base_s: f64,
) -> Result<FusedSetupAndProveResult, Box<dyn Error>> {
    let total_start = Instant::now();
    initialize_stage_trace(&status_path)?;

    // Load the request-specific witness and compiled circuit exactly once.
    let data = GraphWitness::from_path(witness_path)?;
    let mut circuit = GraphCircuit::load(compiled_circuit_path)?;
    circuit.load_graph_witness(&data)?;

    let pretty_public_inputs = circuit.pretty_public_inputs(&data)?;
    let public_inputs = circuit.prepare_public_inputs(&data)?;
    let logrows = circuit.settings().run_args.logrows;
    let proof_split_commits: Option<ProofSplitCommit> = data.into();
    publish_stage(&status_path, stage_trace_elapsed_base_s, &total_start, "KEY_GEN")?;
    let keygen_start = Instant::now();

    append_stage_trace(
        &status_path,
        stage_trace_elapsed_base_s,
        &total_start,
        "LOAD_PARAMS",
    )?;
    // Load the SRS once and reuse it for both key generation and proof
    // generation.
    let params = crate::execute::load_params_prover::<KZGCommitmentScheme<Bn256>>(
        srs_path,
        logrows,
    )?;

    let empty_circuit = <GraphCircuit as Circuit<Fr>>::without_witnesses(&circuit);
    append_stage_trace(
        &status_path,
        stage_trace_elapsed_base_s,
        &total_start,
        "KEYGEN_VK",
    )?;
    let vk_start = Instant::now();
    let vk = keygen_vk_custom(
        &params,
        &empty_circuit,
        !disable_selector_compression,
    )?;
    let vk_time_s = vk_start.elapsed().as_secs_f64();

    append_stage_trace(
        &status_path,
        stage_trace_elapsed_base_s,
        &total_start,
        "KEYGEN_PK",
    )?;
    let pk_start = Instant::now();
    let pk = keygen_pk(&params, vk, &empty_circuit)?;
    let pk_time_s = pk_start.elapsed().as_secs_f64();
    write_keygen_times(vk_time_s, pk_time_s)?;
    let keygen_time_s = keygen_start.elapsed().as_secs_f64();

    publish_stage(
        &status_path,
        stage_trace_elapsed_base_s,
        &total_start,
        "PROVING",
    )?;
    let prove_start = Instant::now();
    let mut snark = create_proof_circuit::<
        KZGCommitmentScheme<Bn256>,
        _,
        ProverSHPLONK<_>,
        VerifierSHPLONK<_>,
        KZGSingleStrategy<_>,
        _,
        EvmTranscript<_, _, _, _>,
        EvmTranscript<_, _, _, _>,
    >(
        circuit,
        vec![public_inputs],
        &params,
        &pk,
        check_mode,
        proof_split_commits,
        None,
    )?;
    let prove_time_s = prove_start.elapsed().as_secs_f64();

    publish_stage(
        &status_path,
        stage_trace_elapsed_base_s,
        &total_start,
        "PERSISTING_KEYS",
    )?;
    let persist_start = Instant::now();
    save_vk::<G1Affine>(&vk_path, pk.get_vk())?;
    save_pk::<G1Affine>(&pk_path, &pk)?;
    let key_persist_time_s = persist_start.elapsed().as_secs_f64();

    snark.pretty_public_inputs = pretty_public_inputs;

    publish_stage(
        &status_path,
        stage_trace_elapsed_base_s,
        &total_start,
        "PERSISTING_PROOF",
    )?;
    let proof_persist_start = Instant::now();
    if let Some(proof_path) = proof_path {
        snark.save(&proof_path)?;
    }
    let proof_persist_time_s = proof_persist_start.elapsed().as_secs_f64();

    Ok(FusedSetupAndProveResult {
        keygen_time_s,
        prove_time_s,
        proof_persist_time_s,
        key_persist_time_s,
        total_time_s: total_start.elapsed().as_secs_f64(),
    })
}


#[cfg(feature = "python-bindings")]
#[pyfunction(name = "setup_and_prove", signature = (
    witness, model, vk_path, pk_path, proof_path=None, srs_path=None,
    proof_type="single", disable_selector_compression=false,
    status_path=None, stage_trace_elapsed_base_s=0.0,
))]
pub(crate) fn py_setup_and_prove(
    py: Python<'_>, witness: PathBuf, model: PathBuf, vk_path: PathBuf,
    pk_path: PathBuf, proof_path: Option<PathBuf>, srs_path: Option<PathBuf>,
    proof_type: &str, disable_selector_compression: bool,
    status_path: Option<PathBuf>, stage_trace_elapsed_base_s: f64,
) -> PyResult<Py<PyAny>> {
    if proof_type != "single" {
        return Err(PyValueError::new_err("Only single KZG proofs are supported upstream"));
    }
    let result = setup_and_prove(
        witness, model, vk_path, pk_path, proof_path, srs_path,
        CheckMode::UNSAFE, disable_selector_compression,
        status_path, stage_trace_elapsed_base_s,
    ).map_err(|e| PyRuntimeError::new_err(format!("Failed fused setup+prove: {e}")))?;
    let output = PyDict::new(py);
    output.set_item("keygen_time_s", result.keygen_time_s)?;
    output.set_item("prove_time_s", result.prove_time_s)?;
    output.set_item("proof_persist_time_s", result.proof_persist_time_s)?;
    output.set_item("key_persist_time_s", result.key_persist_time_s)?;
    output.set_item("total_time_s", result.total_time_s)?;
    Ok(output.into_any().unbind())
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn _ezkl_fused(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("stage_trace_version", STAGE_TRACE_VERSION)?;
    module.add_function(wrap_pyfunction!(py_setup_and_prove, module)?)?;
    Ok(())
}
