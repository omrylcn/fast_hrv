//! Cost of `frequency_domain_lomb` on a 5 min window (≈ 350 beats): `cargo run --release --example lomb_cost`
use fast_hrv::{FrequencyOptions, frequency_domain_lomb, frequency_domain_with};
use std::time::Instant;

fn main() {
    let (mut t, mut nn, mut s) = (Vec::new(), Vec::new(), 0.0);
    while s < 300.0 {
        let rr = 0.85 + 0.03 * (2.0 * std::f64::consts::PI * 0.25 * s).sin();
        s += rr;
        t.push(s);
        nn.push(rr * 1000.0);
    }
    let o = FrequencyOptions::default();
    let n = 200;
    let t0 = Instant::now();
    let mut acc = 0.0;
    for _ in 0..n {
        acc += frequency_domain_lomb(&t, &nn, &o).hf;
    }
    let lomb = t0.elapsed().as_secs_f64() / n as f64;
    let t0 = Instant::now();
    for _ in 0..n {
        acc += frequency_domain_with(&nn, &o).hf;
    }
    let welch = t0.elapsed().as_secs_f64() / n as f64;
    println!(
        "{} beats: lomb {:.0} µs, welch {:.0} µs per 5 min window ({acc:.0})",
        t.len(),
        lomb * 1e6,
        welch * 1e6
    );
}
