//! Baevsky stress index: hand-computed values (Kubios definitions; no NeuroKit2 reference exists).

use fast_hrv::time_domain;

#[test]
fn hand_computed_value() {
    // 10 intervals: 800 ×6 (bin 800–850), 860 ×2, 900, 1000 ms
    let nn = [
        800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 860.0, 860.0, 900.0, 1000.0,
    ];
    let t = time_domain(&nn);
    // Mo = median = 800 ms = 0.8 s; AMo = 6 / 10 = 60 %; MxDMn = 1000 − 800 = 0.2 s
    let si = 60.0 / (2.0 * 0.8 * 0.2);
    assert!((t.stress_index - si).abs() < 1e-9, "{}", t.stress_index);
    assert!((t.stress_index - 187.5).abs() < 1e-9);
    assert!((t.stress_index_sqrt - si.sqrt()).abs() < 1e-9);
}

#[test]
fn rigid_rhythm_scores_higher_than_variable_rhythm() {
    // same mean, narrow spread vs wide spread
    let narrow: Vec<f64> = (0..300)
        .map(|i| 850.0 + 10.0 * ((i as f64) * 0.7).sin())
        .collect();
    let wide: Vec<f64> = (0..300)
        .map(|i| 850.0 + 80.0 * ((i as f64) * 0.7).sin())
        .collect();
    let (a, b) = (time_domain(&narrow), time_domain(&wide));
    assert!(
        a.stress_index > 5.0 * b.stress_index,
        "{} vs {}",
        a.stress_index,
        b.stress_index
    );
}

#[test]
fn degenerate_inputs_give_nan() {
    assert!(
        time_domain(&[800.0; 10]).stress_index.is_nan(),
        "zero spread"
    );
    assert!(
        time_domain(&[800.0, 810.0]).stress_index.is_nan(),
        "too short"
    );
    let mut nn = vec![800.0; 20];
    nn[3] = f64::NAN;
    assert!(time_domain(&nn).stress_index.is_nan());
}

#[test]
fn one_artifact_changes_it_strongly() {
    // documents the known weakness: MxDMn is set by a single outlier
    let clean: Vec<f64> = (0..300)
        .map(|i| 850.0 + 30.0 * ((i as f64) * 0.3).sin())
        .collect();
    let mut dirty = clean.clone();
    dirty[100] = 1700.0; // one missed beat
    let (a, b) = (time_domain(&clean), time_domain(&dirty));
    assert!(
        a.stress_index > 5.0 * b.stress_index,
        "{} vs {}",
        a.stress_index,
        b.stress_index
    );
}
