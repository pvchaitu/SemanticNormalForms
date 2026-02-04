{
  "EENF": {
    "tau": 0.01,                          // Set to 95th percentile of stable-entity variance; allows small model noise but flags unstable entities.
    "calibration_quantile": 0.95         // Use high quantile to tolerate normal stochasticity while catching outliers.
  },
  "AANF": {
    "tau": 0.9,                          // High similarity threshold for aliasing; calibrated on labeled alias pairs to avoid false merges.
    "calibration_quantile": 0.99         // Use very conservative quantile for AANF to reduce accidental aliasing across domains.
  },
  "CMNF": {
    "tau": 0.05,                         // Max allowable average inner product across contexts; keeps context leakage low in practice.
    "calibration_quantile": 0.95
  },
  "DBNF": {
    "tau": 0.25,                         // Per-attribute drift threshold (L2 on normalized embeddings). Chosen to balance sensitivity and robustness.
    "calibration_quantile": 0.90
  },
  "ECNF": {
    "m_min": 3,                          // Minimum distinct evidence items (or sources) required for a safe auto-merge.
    "score_threshold": 0.55,             // Aggregate evidence score threshold for permissive auto-merge when count >= m_min.
    "strong_score_threshold": 0.72       // Strong aggregate score that can override low count (trusted signals).
  },
  "RRNF": {
    "tau": 0.7                           // Role-consistency threshold; tuned to accept common role chains while flagging incompatible inferences.
  },
  "PONF": {
    "tau": 0.1                           // Partition overlap threshold; keeps large semantic partitions distinct.
  },
  "decision_policy": {
    "dbnf_fail_fraction": 0.5,           // Require >50% attributes failing DBNF AND median drift high to block deployment.
    "dbnf_median_factor": 2.0,           // Median drift must exceed (factor * tau) to be considered systemic.
    "auto_merge_min_score": 0.80,        // For fully automatic merges (no human review), require very strong aggregate evidence.
    "auto_merge_min_count": 4           // And require evidence from at least this many distinct signals/sources for auto-merge.
  }
}
