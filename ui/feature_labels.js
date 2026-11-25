// Display names for the analysis feature names returned by /api/dataset/inspect and
// /api/analytics (src/analytics/dataset_analysis.py). Shared by the Dataset and Analytics pages
// so both label every feature identically; unknown names (dataset/synthetic features) fall
// back to the raw column name.
const FEATURE_LABELS = {
    frequency_mhz: "Carrier Frequency (MHz)",
    pulse_width: "Pulse Width (µs)",
    pri_us: "PRI (µs)",
    rise_time_ns: "Rise Time (ns)",
    fall_time_ns: "Fall Time (ns)",
    bandwidth_hz: "Bandwidth (Hz)",
    duty_cycle_pct: "Duty Cycle (%)",
    prf_hz: "PRF (Hz)",
    pw_x_pri: "PW × PRI (µs²)",
    tb_product: "Time-Bandwidth Product (PW × BW)",
    snr: "SNR (dB)",
    Modlnval: "Modulation Value",
};
const featureLabel = (k) => FEATURE_LABELS[k] || k;

// Compact number formatting for stats tiles and tables.
const fmtNum = (v) => {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(2);
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2);
    return v.toFixed(3);
};
