//! Accelerometer context for an HRV window: how much the person moved and whether posture changed.
//!
//! HRV depends on posture and physical activity at least as much as on stress or fatigue, so a value
//! is only interpretable with this context. The library supplies the measurements; thresholds that
//! turn them into decisions (gating, "after activity" periods) belong to the application, so all of
//! them are parameters here.
//!
//! - **Magnitude** `|a|` (g) per sample; its mean and standard deviation over the window.
//! - **ENMO** (Euclidean norm minus one, negatives set to 0; van Hees et al. 2013), a common activity
//!   measure that is ~0 at rest.
//! - **Activity level** from the standard deviation of `|a|`: below `rest_sd_g` rest, below
//!   `active_sd_g` light, else active (defaults 0.02 g / 0.06 g are conventions for a chest sensor at
//!   ~1 Hz, not validated thresholds).
//! - **Gravity direction**: the unit vector of the mean acceleration (at rest it points down in the
//!   sensor's frame). With a reference vector recorded in a known posture (e.g. upright standing),
//!   **tilt** is the angle between them. Without calibration only changes are meaningful.
//! - **Posture change**: angle between the mean vectors of the first and second half of the window;
//!   above `posture_change_deg` the window is flagged. HRV needs time to settle after a posture
//!   change (a 2024 study trimmed 30 s at each end of a posture episode); how long to wait is the
//!   application's choice.

/// Thresholds (all conventions; see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MotionConfig {
    pub rest_sd_g: f64,
    pub active_sd_g: f64,
    pub posture_change_deg: f64,
    /// Fewer samples than this give [`Activity::Unknown`] and no posture result.
    pub min_samples: usize,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            rest_sd_g: 0.02,
            active_sd_g: 0.06,
            posture_change_deg: 30.0,
            min_samples: 5,
        }
    }
}

/// Activity level of a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum Activity {
    Rest,
    Light,
    Active,
    /// Too few samples.
    Unknown,
}

/// Accelerometer summary of one window.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MotionContext {
    pub n_samples: usize,
    pub magnitude_mean_g: f64,
    pub magnitude_sd_g: f64,
    pub enmo_mean_g: f64,
    pub activity: Activity,
    /// Unit vector of the mean acceleration (NaN when unknown).
    pub gravity: [f64; 3],
    /// Angle (deg) between `gravity` and the reference vector, if one was given.
    pub tilt_deg: Option<f64>,
    /// Angle (deg) between the mean vectors of the first and second half of the window.
    pub posture_shift_deg: f64,
    pub posture_change: bool,
}

/// Angle between two vectors (deg); NaN if either is zero.
pub fn angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if na == 0.0 || nb == 0.0 {
        return f64::NAN;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0).acos().to_degrees()
}

fn mean_vec(xyz: &[[f64; 3]]) -> [f64; 3] {
    let n = xyz.len().max(1) as f64;
    let mut m = [0.0; 3];
    for v in xyz {
        for k in 0..3 {
            m[k] += v[k] / n;
        }
    }
    m
}

/// [`MotionContext`] of the samples of one window (g, sensor frame). `reference`: gravity vector in
/// a known posture for [`MotionContext::tilt_deg`].
pub fn motion_context(
    xyz: &[[f64; 3]],
    reference: Option<[f64; 3]>,
    cfg: &MotionConfig,
) -> MotionContext {
    let samples: Vec<[f64; 3]> = xyz
        .iter()
        .copied()
        .filter(|v| v.iter().all(|c| c.is_finite()))
        .collect();
    let n = samples.len();
    let nan = f64::NAN;
    if n < cfg.min_samples.max(2) {
        return MotionContext {
            n_samples: n,
            magnitude_mean_g: nan,
            magnitude_sd_g: nan,
            enmo_mean_g: nan,
            activity: Activity::Unknown,
            gravity: [nan; 3],
            tilt_deg: None,
            posture_shift_deg: nan,
            posture_change: false,
        };
    }
    let mags: Vec<f64> = samples
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .collect();
    let mean = mags.iter().sum::<f64>() / n as f64;
    let sd = (mags.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    let enmo = mags.iter().map(|m| (m - 1.0).max(0.0)).sum::<f64>() / n as f64;
    let activity = if sd < cfg.rest_sd_g {
        Activity::Rest
    } else if sd < cfg.active_sd_g {
        Activity::Light
    } else {
        Activity::Active
    };
    let g = mean_vec(&samples);
    let gn = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    let gravity = if gn > 0.0 {
        [g[0] / gn, g[1] / gn, g[2] / gn]
    } else {
        [nan; 3]
    };
    let shift = angle_deg(mean_vec(&samples[..n / 2]), mean_vec(&samples[n / 2..]));
    MotionContext {
        n_samples: n,
        magnitude_mean_g: mean,
        magnitude_sd_g: sd,
        enmo_mean_g: enmo,
        activity,
        gravity,
        tilt_deg: reference.map(|r| angle_deg(g, r)),
        posture_shift_deg: shift,
        posture_change: shift > cfg.posture_change_deg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rest_activity_and_posture() {
        let cfg = MotionConfig::default();
        // still, upright (gravity along −y)
        let still: Vec<[f64; 3]> = (0..60).map(|_| [0.0, -1.0, 0.0]).collect();
        let c = motion_context(&still, Some([0.0, -1.0, 0.0]), &cfg);
        assert_eq!(c.activity, Activity::Rest);
        assert!(c.enmo_mean_g.abs() < 1e-12 && c.tilt_deg.unwrap() < 1e-6);
        assert!(!c.posture_change);
        // walking-like bounce of ±0.2 g
        let walk: Vec<[f64; 3]> = (0..60)
            .map(|i| [0.0, -1.0 - 0.2 * (i as f64 * 2.0).sin(), 0.0])
            .collect();
        assert_eq!(motion_context(&walk, None, &cfg).activity, Activity::Active);
        // lying down halfway: gravity turns 90°
        let mut lie = still.clone();
        for v in lie.iter_mut().skip(30) {
            *v = [0.0, 0.0, -1.0];
        }
        let c = motion_context(&lie, Some([0.0, -1.0, 0.0]), &cfg);
        assert!(c.posture_change, "{c:?}");
        assert!((c.posture_shift_deg - 90.0).abs() < 1e-6);
        // too few samples
        assert_eq!(
            motion_context(&still[..3], None, &cfg).activity,
            Activity::Unknown
        );
        assert!((angle_deg([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) - 90.0).abs() < 1e-12);
    }
}
