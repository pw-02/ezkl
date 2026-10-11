use crate::circuit::CheckMode;
use crate::graph::{GraphCircuit, GraphWitness};
use crate::pfsys::evm::aggregation_kzg::PoseidonTranscript;
use crate::pfsys::{
    create_keys, create_proof_circuit, save_pk, save_vk, ProofSplitCommit, ProofType, Snark,
    StrategyType, TranscriptType,
};
use crate::Commitments;
use halo2_proofs::poly::ipa::commitment::IPACommitmentScheme;
use halo2_proofs::poly::ipa::multiopen::{ProverIPA, VerifierIPA};
use halo2_proofs::poly::ipa::strategy::AccumulatorStrategy as IPAAccumulatorStrategy;
use halo2_proofs::poly::ipa::strategy::SingleStrategy as IPASingleStrategy;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::AccumulatorStrategy as KZGAccumulatorStrategy;
use halo2_proofs::poly::kzg::strategy::SingleStrategy as KZGSingleStrategy;
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use snark_verifier::loader::native::NativeLoader;
use snark_verifier::system::halo2::compile;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use snark_verifier::system::halo2::Config;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "python-bindings")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;

/// Timing information returned by the fused cold path.
#[derive(Debug)]
pub(crate) struct FusedSetupAndProveResult {
    pub snark: Snark<Fr, G1Affine>,
    /// Includes loading prover parameters/SRS and generating VK + PK.
    pub keygen_time_s: f64,
    /// Time spent creating the proof, excluding proof serialization.
    pub prove_time_s: f64,
    pub proof_persist_time_s: f64,
    pub key_persist_time_s: f64,
    pub total_time_s: f64,
}

fn publish_stage(status_path: &Option<PathBuf>, stage: &str) -> Result<(), Box<dyn Error>> {
    if let Some(path) = status_path {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, stage)?;
    }
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
    proof_type: ProofType,
    check_mode: CheckMode,
    disable_selector_compression: bool,
    status_path: Option<PathBuf>,
) -> Result<FusedSetupAndProveResult, Box<dyn Error>> {
    let total_start = Instant::now();

    // Load the request-specific witness and compiled circuit exactly once.
    let data = GraphWitness::from_path(witness_path)?;
    let mut circuit = GraphCircuit::load(compiled_circuit_path)?;
    circuit.load_graph_witness(&data)?;

    let pretty_public_inputs = circuit.pretty_public_inputs(&data)?;
    let public_inputs = circuit.prepare_public_inputs(&data)?;
    let circuit_settings = circuit.settings().clone();
    let proof_split_commits: Option<ProofSplitCommit> = data.into();
    let commitment: Commitments = circuit_settings.run_args.commitment.into();
    let logrows = circuit_settings.run_args.logrows;
    let strategy: StrategyType = proof_type.into();
    let transcript: TranscriptType = proof_type.into();

    publish_stage(&status_path, "KEY_GEN")?;
    let keygen_start = Instant::now();

    let (mut snark, keygen_time_s, phase_times) = match commitment {
        Commitments::KZG => {
            // Load the SRS once and reuse it for both key generation and proof
            // generation.
            let params = crate::execute::load_params_prover::<KZGCommitmentScheme<Bn256>>(
                srs_path,
                logrows,
                Commitments::KZG,
            )?;
            let pk = create_keys::<KZGCommitmentScheme<Bn256>, GraphCircuit>(
                &circuit,
                &params,
                disable_selector_compression,
            )?;
            let keygen_time_s = keygen_start.elapsed().as_secs_f64();

            publish_stage(&status_path, "PROVING")?;
            let prove_start = Instant::now();
            let snark = match strategy {
                StrategyType::Single => create_proof_circuit::<
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
                    commitment,
                    transcript,
                    proof_split_commits,
                    None,
                )?,
                StrategyType::Accum => {
                    let protocol = Some(compile(
                        &params,
                        pk.get_vk(),
                        Config::kzg().with_num_instance(vec![public_inputs.len()]),
                    ));
                    create_proof_circuit::<
                        KZGCommitmentScheme<Bn256>,
                        _,
                        ProverSHPLONK<_>,
                        VerifierSHPLONK<_>,
                        KZGAccumulatorStrategy<_>,
                        _,
                        PoseidonTranscript<NativeLoader, _>,
                        PoseidonTranscript<NativeLoader, _>,
                    >(
                        circuit,
                        vec![public_inputs],
                        &params,
                        &pk,
                        check_mode,
                        commitment,
                        transcript,
                        proof_split_commits,
                        protocol,
                    )?
                }
            };
            let prove_time_s = prove_start.elapsed().as_secs_f64();

            publish_stage(&status_path, "PERSISTING_KEYS")?;
            let persist_start = Instant::now();
            save_vk::<G1Affine>(&vk_path, pk.get_vk())?;
            save_pk::<G1Affine>(&pk_path, &pk)?;
            let key_persist_time_s = persist_start.elapsed().as_secs_f64();

            (snark, keygen_time_s, (prove_time_s, key_persist_time_s))
        }
        Commitments::IPA => {
            let params = crate::execute::load_params_prover::<IPACommitmentScheme<G1Affine>>(
                srs_path,
                logrows,
                Commitments::IPA,
            )?;
            let pk = create_keys::<IPACommitmentScheme<G1Affine>, GraphCircuit>(
                &circuit,
                &params,
                disable_selector_compression,
            )?;
            let keygen_time_s = keygen_start.elapsed().as_secs_f64();

            publish_stage(&status_path, "PROVING")?;
            let prove_start = Instant::now();
            let snark = match strategy {
                StrategyType::Single => create_proof_circuit::<
                    IPACommitmentScheme<G1Affine>,
                    _,
                    ProverIPA<_>,
                    VerifierIPA<_>,
                    IPASingleStrategy<_>,
                    _,
                    EvmTranscript<_, _, _, _>,
                    EvmTranscript<_, _, _, _>,
                >(
                    circuit,
                    vec![public_inputs],
                    &params,
                    &pk,
                    check_mode,
                    commitment,
                    transcript,
                    proof_split_commits,
                    None,
                )?,
                StrategyType::Accum => {
                    let protocol = Some(compile(
                        &params,
                        pk.get_vk(),
                        Config::ipa().with_num_instance(vec![public_inputs.len()]),
                    ));
                    create_proof_circuit::<
                        IPACommitmentScheme<G1Affine>,
                        _,
                        ProverIPA<_>,
                        VerifierIPA<_>,
                        IPAAccumulatorStrategy<_>,
                        _,
                        PoseidonTranscript<NativeLoader, _>,
                        PoseidonTranscript<NativeLoader, _>,
                    >(
                        circuit,
                        vec![public_inputs],
                        &params,
                        &pk,
                        check_mode,
                        commitment,
                        transcript,
                        proof_split_commits,
                        protocol,
                    )?
                }
            };
            let prove_time_s = prove_start.elapsed().as_secs_f64();

            publish_stage(&status_path, "PERSISTING_KEYS")?;
            let persist_start = Instant::now();
            save_vk::<G1Affine>(&vk_path, pk.get_vk())?;
            save_pk::<G1Affine>(&pk_path, &pk)?;
            let key_persist_time_s = persist_start.elapsed().as_secs_f64();

            (snark, keygen_time_s, (prove_time_s, key_persist_time_s))
        }
    };

    let (prove_time_s, key_persist_time_s) = phase_times;
    snark.pretty_public_inputs = pretty_public_inputs;

    publish_stage(&status_path, "PERSISTING_PROOF")?;
    let proof_persist_start = Instant::now();
    if let Some(proof_path) = proof_path {
        snark.save(&proof_path)?;
    }
    let proof_persist_time_s = proof_persist_start.elapsed().as_secs_f64();

    Ok(FusedSetupAndProveResult {
        snark,
        keygen_time_s,
        prove_time_s,
        proof_persist_time_s,
        key_persist_time_s,
        total_time_s: total_start.elapsed().as_secs_f64(),
    })
}

/// Python bridge for the fused path.
///
/// The main `ezkl` extension module predates this experiment. To avoid a large,
/// fragile rewrite of that binding file, the same shared library also exports
/// a tiny `_ezkl_fused` module. zkInfer loads this module from `ezkl.__file__`
/// and calls this function only on a setup-cache miss.
#[cfg(feature = "python-bindings")]
#[pyfunction(name = "setup_and_prove", signature = (
    witness,
    model,
    vk_path,
    pk_path,
    proof_path=None,
    srs_path=None,
    proof_type=ProofType::default(),
    disable_selector_compression=false,
    status_path=None,
))]
fn py_setup_and_prove(
    py: Python<'_>,
    witness: PathBuf,
    model: PathBuf,
    vk_path: PathBuf,
    pk_path: PathBuf,
    proof_path: Option<PathBuf>,
    srs_path: Option<PathBuf>,
    proof_type: ProofType,
    disable_selector_compression: bool,
    status_path: Option<PathBuf>,
) -> PyResult<PyObject> {
    let result = setup_and_prove(
        witness,
        model,
        vk_path,
        pk_path,
        proof_path,
        srs_path,
        proof_type,
        CheckMode::UNSAFE,
        disable_selector_compression,
        status_path,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed fused setup+prove: {e}")))?;

    // The proof and keys are persisted by Rust. Returning the full Snark here
    // would hex-encode/copy the proof back through Python for no serving benefit.
    let output = PyDict::new(py);
    output.set_item("keygen_time_s", result.keygen_time_s)?;
    output.set_item("prove_time_s", result.prove_time_s)?;
    output.set_item("key_persist_time_s", result.key_persist_time_s)?;
    output.set_item("proof_persist_time_s", result.proof_persist_time_s)?;
    output.set_item("total_time_s", result.total_time_s)?;
    Ok(output.to_object(py))
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn _ezkl_fused(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_setup_and_prove, module)?)?;
    Ok(())
}
