// Run details window (Training page): everything about one training run — configuration,
// results, detailed evaluation, and how the trained model works (formulas, input scaling,
// feature influence, weights) — plus save / use / download actions.
//
//   openRunDetails(runId)       — opens the window
//   window.onRunDetailsChanged  — optional callback after save/unsave/activate/delete

(function () {
    const css = `
    .rd-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.65); display: none; z-index: 1000; align-items: flex-start; justify-content: center; overflow-y: auto; padding: 30px 12px; }
    .rd-overlay.open { display: flex; }
    .rd-panel { background: var(--panel-bg, #111827); border: 1px solid var(--border-color, #1f2937); border-radius: 8px; width: min(1150px, 100%); padding: 18px 22px; position: relative; }
    .rd-close { position: absolute; top: 10px; right: 14px; background: none; border: none; color: #9ca3af; font-size: 22px; cursor: pointer; width: auto; margin: 0; padding: 0 6px; }
    .rd-close:hover { color: white; }
    .rd-title { font-size: 13pt; font-weight: bold; color: var(--accent-blue, #00d2ff); margin-right: 40px; }
    .rd-sub { font-size: 8.5pt; color: #9ca3af; margin: 4px 0 12px; }
    .rd-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 8pt; font-weight: bold; margin-left: 6px; vertical-align: middle; }
    .rd-badge.saved { background: rgba(245,158,11,0.2); color: #fbbf24; }
    .rd-badge.active { background: rgba(16,185,129,0.2); color: #34d399; }
    .rd-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .rd-actions button { width: auto; margin-top: 0; padding: 7px 12px; font-size: 8.5pt; }
    .rd-tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border-color, #1f2937); margin-bottom: 14px; }
    .rd-tab { width: auto; margin: 0; background: none; color: #9ca3af; border: none; border-bottom: 2px solid transparent; padding: 8px 14px; font-size: 9pt; cursor: pointer; border-radius: 0; }
    .rd-tab.active { color: var(--accent-blue, #00d2ff); border-bottom-color: var(--accent-blue, #00d2ff); font-weight: bold; }
    .rd-section { margin-bottom: 20px; }
    .rd-h { font-size: 9.5pt; font-weight: bold; color: var(--accent-blue, #00d2ff); text-transform: uppercase; margin-bottom: 6px; }
    .rd-formula { background: #0b1220; border: 1px solid #1f2937; border-radius: 6px; padding: 10px 12px; margin-bottom: 10px; }
    .rd-formula .t { font-size: 8.5pt; color: #fbbf24; font-weight: bold; margin-bottom: 4px; }
    .rd-formula pre { margin: 0; font-family: Consolas, 'Courier New', monospace; font-size: 8.8pt; color: #e5e7eb; white-space: pre-wrap; line-height: 1.55; }
    .rd-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-bottom: 14px; }
    .rd-kv { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 2px 18px; font-size: 8.8pt; }
    .rd-kv div { display: flex; justify-content: space-between; border-bottom: 1px dashed #1f2937; padding: 3px 0; }
    .rd-kv span { color: #9ca3af; } .rd-kv b { color: #e5e7eb; text-align: right; }
    .rd-cm td, .rd-cm th { text-align: center !important; padding: 4px 6px !important; font-size: 8pt; }
    .rd-scroll { overflow-x: auto; }
    .rd-wtable td { font-family: Consolas, monospace; font-size: 8pt; text-align: right; }
    `;
    const style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);

    const esc = escapeHtml;  // ui/escape.js
    const num = (v, d = 4) => (v === null || v === undefined) ? '—' : (typeof fmtNum === 'function' ? fmtNum(v) : Number(v).toFixed(d));
    const flabel = (f) => (typeof featureLabel === 'function') ? featureLabel(f) : f;
    let charts = [];
    let current = null;

    function overlay() {
        let el = document.getElementById('rd-overlay');
        if (!el) {
            el = document.createElement('div');
            el.id = 'rd-overlay';
            el.className = 'rd-overlay';
            el.innerHTML = '<div class="rd-panel"><button class="rd-close" title="Close">&times;</button><div id="rd-body"></div></div>';
            el.addEventListener('click', (e) => { if (e.target === el) closeRunDetails(); });
            el.querySelector('.rd-close').onclick = closeRunDetails;
            document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeRunDetails(); });
            document.body.appendChild(el);
        }
        return el;
    }

    function destroyCharts() { charts.forEach(c => c.destroy()); charts = []; }

    window.closeRunDetails = function () {
        destroyCharts();
        const el = document.getElementById('rd-overlay');
        if (el) el.classList.remove('open');
    };

    window.openRunDetails = async function (runId, tab = 'overview') {
        const el = overlay();
        el.classList.add('open');
        document.getElementById('rd-body').innerHTML = '<div class="hint">Loading…</div>';
        const data = await (await fetch(`/api/runs/${runId}`)).json();
        if (data.status !== 'success') { document.getElementById('rd-body').innerHTML = `<div class="hint">${esc(data.message)}</div>`; return; }
        current = data;
        render(tab);
    };

    function render(tab) {
        destroyCharts();
        const { run: r, report, has_model, is_active } = current;
        const reportOk = report && !report.error;
        const tabs = [['overview', 'Overview'], ['evaluation', 'Evaluation'], ['explain', 'Explainability (formulas & weights)']];
        document.getElementById('rd-body').innerHTML = `
            <div class="rd-title">Training run #${r.id}
                ${r.saved_name ? `<span class="rd-badge saved">★ ${esc(r.saved_name)}</span>` : ''}
                ${is_active ? '<span class="rd-badge active">● In use</span>' : ''}</div>
            <div class="rd-sub">Trained ${esc(r.timestamp_ist)}${r.saved_at_ist ? ` · saved ${esc(r.saved_at_ist)}` : ''} · ${esc(r.backbone_type)} backbone · ${esc((r.features || 'features not recorded').split(',').join(', '))}</div>
            <div class="rd-actions">
                ${has_model ? `
                    <button onclick="rdSave()" style="background: var(--accent-purple, #7c3aed); color: white;">${r.saved_name ? 'Rename' : '★ Save Model…'}</button>
                    ${r.saved_name ? '<button onclick="rdUnsave()" style="background:#374151; color:white;">Unsave</button>' : ''}
                    ${is_active ? '' : '<button onclick="rdActivateSeq()">Use for Inference / Live Simulation</button>'}
                    <button onclick="location.href='/api/runs/${r.id}/download'" style="background:#374151; color:white;">⬇ Download Model Package (.zip)</button>
                    <button onclick="location.href='/api/runs/${r.id}/weights'" style="background:#374151; color:white;">⬇ All Weights (JSON)</button>`
                : '<div class="hint">This run\'s trained model wasn\'t kept (it predates saved models, or was pruned as an old unsaved run), so it can\'t be saved, used or downloaded. Configuration and results are shown below.</div>'}
                <button onclick="rdDelete()" style="background:#b91c1c; color:white;">🗑 Delete Run</button>
            </div>
            <div id="rd-msg" class="hint" style="margin-bottom:8px;">Pulse-sequence model: classifies whole tracks. The model in use serves the Inference and Live Simulation pages; any kept model can be chosen on the Testing page.</div>
            <div class="rd-tabs">${tabs.map(([id, t]) => `<button class="rd-tab ${id === tab ? 'active' : ''}" onclick="rdTab('${id}')">${t}</button>`).join('')}</div>
            <div id="rd-tab-body"></div>`;
        const body = document.getElementById('rd-tab-body');
        if (tab === 'overview') renderOverview(body, r, report);
        else if (!reportOk) body.innerHTML = `<div class="hint">${report && report.error ? esc(report.error) : 'Detailed evaluation and explainability are recorded for runs trained after this feature was added; this run predates it.'}</div>`;
        else if (tab === 'evaluation') { renderEvaluation(body, r, report.evaluation); if (report.sequence) renderSequenceEval(body, report.sequence); }
        else renderExplain(body, r, report);
    }

    window.rdTab = (t) => render(t);

    function chart(canvasId, config) {
        const c = new Chart(document.getElementById(canvasId).getContext('2d'), config);
        charts.push(c);
        return c;
    }

    function renderOverview(body, r, report) {
        const ev = report && report.evaluation;
        const tile = (v, l) => `<div class="metric-block"><div class="metric-value" style="font-size:16pt;">${v}</div><div class="metric-label">${l}</div></div>`;
        const kv = (k, v) => `<div><span>${k}</span><b>${v ?? '—'}</b></div>`;
        const pct = (x) => x === null || x === undefined ? '—' : `${(x * 100).toFixed(1)}%`;
        body.innerHTML = `
            <div class="rd-grid">
                ${tile(`${(r.closed_set_accuracy * 100).toFixed(1)}%`, 'Closed-set accuracy')}
                ${tile(r.open_set_auroc !== null ? Number(r.open_set_auroc).toFixed(3) : 'N/A', 'Open-set AUROC (threshold)')}
                ${tile(r.open_set_auroc_rpl !== null && r.open_set_auroc_rpl !== undefined ? Number(r.open_set_auroc_rpl).toFixed(3) : 'N/A', `Open-set AUROC (${r.rpl_adversarial ? 'ARPL' : 'RPL'})`)}
                ${tile(ev && ev.macro_f1 !== null ? ev.macro_f1.toFixed(3) : '—', 'Macro F1 (known classes)')}
                ${tile(num(r.final_loss), 'Final training loss')}
            </div>
            <div class="rd-section"><div class="rd-h">Configuration</div><div class="rd-kv">
                ${kv('Input', r.window_mode === 'track' ? `Pulse sequence — whole track (its first ${r.window_k} pulses, one input per track)` : `Pulse sequence — windows of ${r.window_k} pulses`)}
                ${kv('Backbone', esc(r.backbone_type))}
                ${kv('Embedding size (d)', r.embedding_dim || 128)}
                ${kv('Input features', esc((r.features || '—').split(',').join(', ')))}
                ${kv('Distance metric', esc(r.distance_metric))}
                ${kv('Min classifier confidence (τ_c)', r.classifier_confidence_threshold)}
                ${kv('OOD distance threshold (τ_d)', r.ood_distance_threshold)}
                ${report && report.thresholds ? kv('τ_d chosen by', report.thresholds.mode === 'auto'
                    ? `auto: ${(report.thresholds.coverage * 100).toFixed(1)}% of ${report.thresholds.num_windows} known windows within it (${report.thresholds.calibrated_on || 'validation windows'})`
                    : report.thresholds.mode === 'inherited' ? (report.thresholds.calibrated_on || 'inherited from the base model') : 'manual value') : ''}
                ${kv('Epochs / duration', `${r.epochs} / ${r.training_duration_seconds}s`)}
                ${kv('Learning rate', r.learning_rate)}
                ${kv('Triplet margin', r.triplet_margin)}
                ${kv('RPL loss weight / adversarial', `${r.rpl_loss_weight ?? '—'} / ${r.rpl_adversarial ? 'yes' : 'no'}`)}
                ${kv('Classes (known / holdout)', `${r.num_classes - r.num_holdout_classes} / ${r.num_holdout_classes}`)}
                ${kv('Tracks per class / pulses per track', `${r.samples_per_class ?? '—'} / ${r.pulses_per_track ?? '—'}`)}
                ${kv('Missed / spurious pulses / noise tracks', `${pct(r.missed_pct)} / ${pct(r.spurious_pct)} / ${pct(r.noise_pct)}`)}
                ${kv('Dataset', r.dataset_name ?? '(unnamed — made before datasets had names)')}
                ${kv('Dataset seed', r.dataset_seed ?? '—')}
                ${report && report.incremental ? kv('Incremental training', `from run #${report.incremental.parent_run_id}, method ${esc(report.incremental.method)} — ${esc(report.incremental.method_name)}`) : ''}
                ${report && report.incremental ? kv('New families / data used', `${report.incremental.new_class_labels.map(esc).join(', ')} — ${esc(report.incremental.data_used)}`) : ''}
                ${report && report.incremental && report.incremental.stale_prototypes.length ? kv('Prototypes kept from the base model', report.incremental.stale_prototypes.map(esc).join(', ')) : ''}
            </div></div>
            ${renderCrossValidation(report && report.cross_validation)}
            <div class="rd-section"><div class="rd-h">Training loss per epoch</div><div style="height:220px;"><canvas id="rd-loss"></canvas></div></div>`;
        chart('rd-loss', {
            type: 'line',
            data: { labels: r.loss_curve.map((_, i) => i + 1), datasets: [{ label: 'Average loss', data: r.loss_curve, borderColor: '#00d2ff', pointRadius: 2, tension: 0.2 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
                scales: { x: { title: { display: true, text: 'Epoch' } }, y: { title: { display: true, text: 'Loss' } } } },
        });
    }

    // Accuracy vs. pulses observed and per-track results (sequence models), appended to Evaluation.
    function renderSequenceEval(body, sq) {
        const div = document.createElement('div');
        const v = sq.verdicts || {};
        const vrow = (name, o) => o ? `<tr><td>${name}</td>${['VERIFIED_KNOWN_TARGET', 'UNKNOWN_THREAT_ANOMALY', 'OOD_NOISE_REJECTED'].map(k => `<td>${o[k] ?? 0}</td>`).join('')}</tr>` : '';
        div.innerHTML = `<div class="rd-section"><div class="rd-h">Sequence model — accuracy vs. pulses observed</div>
            <div class="hint" style="margin-bottom:6px;">One decision per track after its first 2, 4, 8, … pulses, computed from those pulses only — exactly what Live Simulation decides at that moment (validation tracks for accuracy, plus unknown tracks for AUROC; stored prototypes). Shows how many pulses it needs before the PRI pattern and frequency agility make it sure. Starts at 2 pulses: the first pulse has no PRI.</div>
            <div style="height:240px;"><canvas id="rd-seq-curve"></canvas></div>
            <div class="rd-grid" style="margin-top:10px;">
                <div class="metric-block"><div class="metric-value" style="font-size:16pt;">${sq.track_accuracy !== null ? (sq.track_accuracy * 100).toFixed(1) + '%' : '—'}</div><div class="metric-label">Track accuracy (vote of its windows)</div></div>
                <div class="metric-block"><div class="metric-value" style="font-size:16pt;">${sq.track_unknown_detection !== null && sq.track_unknown_detection !== undefined ? (sq.track_unknown_detection * 100).toFixed(1) + '%' : '—'}</div><div class="metric-label">Unknown tracks flagged</div></div>
                <div class="metric-block"><div class="metric-value" style="font-size:16pt;">${sq.num_tracks_known} / ${sq.num_tracks_unknown}</div><div class="metric-label">Known / unknown tracks</div></div>
                <div class="metric-block"><div class="metric-value" style="font-size:16pt;">${sq.window}</div><div class="metric-label">Window k (pulses)</div></div>
            </div>
            <table class="eda-table" style="max-width:720px; margin-top:8px;"><thead><tr><th>Windows</th><th>Verified known</th><th>Unknown threat</th><th>Rejected</th></tr></thead><tbody>
                ${vrow('Known tracks', v.known)}${vrow('Unknown tracks', v.unknown)}</tbody></table>
            <div class="hint">Track accuracy: each known track's class = most frequent prediction over its windows. Unknown tracks flagged: share of unknown tracks whose windows are mostly not "verified known".</div></div>`;
        body.appendChild(div);
        const c = sq.pulses_curve || [];
        chart('rd-seq-curve', {
            type: 'line',
            data: { labels: c.map(p => p.pulses), datasets: [
                { label: 'Closed-set accuracy', data: c.map(p => p.accuracy), borderColor: '#00d2ff', backgroundColor: '#00d2ff', tension: 0.2 },
                { label: 'Open-set AUROC', data: c.map(p => p.auroc), borderColor: '#f59e0b', backgroundColor: '#f59e0b', tension: 0.2 },
            ] },
            options: { responsive: true, maintainAspectRatio: false,
                scales: { x: { title: { display: true, text: 'Pulses observed (window length)' } }, y: { min: 0, max: 1, title: { display: true, text: 'Score' } } } },
        });
    }

    function renderEvaluation(body, r, ev) {
        const cmMax = Math.max(1, ...ev.confusion_matrix.flat());
        const cell = (v, diag) => {
            const t = v / cmMax;
            const bg = v === 0 ? 'transparent' : diag ? `rgba(16,185,129,${0.2 + 0.8 * t})` : `rgba(239,68,68,${0.25 + 0.75 * t})`;
            return `<td style="background:${bg}; color:${v === 0 ? '#4b5563' : '#fff'};">${v}</td>`;
        };
        const short = (c) => c.replace('RADAR_FAMILY_', 'F');
        const verdictRow = (name, v) => v ? `<tr><td>${name}</td><td>${v.VERIFIED_KNOWN_TARGET}</td><td>${v.UNKNOWN_THREAT_ANOMALY}</td><td>${v.OOD_NOISE_REJECTED}</td></tr>` : '';
        body.innerHTML = `
            <div class="rd-section"><div class="rd-h">Confusion matrix (validation rows, known classes)</div>
                <div class="hint" style="margin-bottom:6px;">Rows = true class, columns = predicted class (F = RADAR_FAMILY_). Green diagonal = correct; red off-diagonal = which classes get confused with which.</div>
                <div class="rd-scroll"><table class="eda-table rd-cm"><thead><tr><th>true \\ pred</th>${ev.classes.map(c => `<th>${short(c)}</th>`).join('')}</tr></thead><tbody>
                ${ev.confusion_matrix.map((row, i) => `<tr><td><b>${short(ev.classes[i])}</b></td>${row.map((v, j) => cell(v, i === j)).join('')}</tr>`).join('')}
                </tbody></table></div></div>
            <div class="rd-section"><div class="rd-h">Per-class precision / recall / F1</div>
                <div class="rd-scroll"><table class="eda-table"><thead><tr><th>Class</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>
                ${ev.per_class.map(p => `<tr><td>${esc(p.class)}</td><td>${p.support}</td><td>${p.precision.toFixed(3)}</td><td>${p.recall.toFixed(3)}</td><td>${p.f1.toFixed(3)}</td></tr>`).join('')}
                </tbody></table></div><div class="hint">Macro F1: ${ev.macro_f1 ?? '—'}</div></div>
            ${ev.roc ? `<div class="rd-section"><div class="rd-h">Open-set ROC curve (prototype distance as the "unknown" score)</div>
                <div class="hint" style="margin-bottom:6px;">Each point is one distance threshold: how many unknown (held-out/noise) rows get flagged (true positive rate) vs. how many known rows get wrongly flagged (false positive rate). AUROC ${r.open_set_auroc ?? '—'} = area under this curve; the diagonal is chance.</div>
                <div style="height:300px; max-width:520px;"><canvas id="rd-roc"></canvas></div></div>` : ''}
            ${ev.verdicts ? `<div class="rd-section"><div class="rd-h">Fusion verdicts at τ_c = ${r.classifier_confidence_threshold}, τ_d = ${r.ood_distance_threshold}</div>
                <table class="eda-table" style="max-width:720px;"><thead><tr><th>Rows</th><th>Verified known</th><th>Unknown threat</th><th>Rejected noise</th></tr></thead><tbody>
                ${verdictRow('Known classes (validation)', ev.verdicts.known)}${verdictRow('Unknown (held-out + noise)', ev.verdicts.unknown)}
                </tbody></table><div class="hint">Ideal: known rows → verified known; unknown rows → unknown threat or rejected.</div></div>` : ''}`;
        if (ev.roc) {
            chart('rd-roc', {
                type: 'scatter',
                data: { datasets: [
                    { label: 'ROC', data: ev.roc.map(p => ({ x: p.fpr, y: p.tpr })), showLine: true, borderColor: '#00d2ff', backgroundColor: '#00d2ff', pointRadius: 0, borderWidth: 2 },
                    { label: 'Chance', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, borderColor: '#6b7280', borderDash: [6, 4], pointRadius: 0 },
                ] },
                options: { responsive: true, maintainAspectRatio: false,
                    scales: { x: { min: 0, max: 1, title: { display: true, text: 'False positive rate (known flagged as unknown)' } },
                              y: { min: 0, max: 1, title: { display: true, text: 'True positive rate (unknown caught)' } } } },
            });
        }
    }

    function renderBranchAblation(ba, r) {
        if (!ba) {
            return (r.backbone_type === 'hybrid' || r.backbone_type === 'seq_hybrid')
                ? `<div class="rd-section"><div class="rd-h">Hybrid branch contribution</div><div class="hint">Not recorded for this run — it was trained before branch analysis existed. Retrain to see it.</div></div>`
                : '';
        }
        const pts = (v) => v === null || v === undefined ? '—' : `${v > 0 ? '−' : v < 0 ? '+' : ''}${Math.abs(v * 100).toFixed(1)} pts`;
        const auDrop = (v) => v === null || v === undefined ? '—' : `${v > 0 ? '−' : v < 0 ? '+' : ''}${Math.abs(v).toFixed(3)}`;
        const s = ba.signal_share;
        return `<div class="rd-section"><div class="rd-h">Hybrid branch contribution (1D-CNN vs Transformer)</div>
            <div class="hint" style="margin-bottom:6px;">The Hybrid model fuses the two branches' 64-number outputs: f = ReLU(W_h · [f_cnn ; f_tf] + b_h). Here each branch is switched off (its 64 numbers set to 0) or neutralised (set to its average over the validation pulses, removing pulse-specific information but keeping its typical level), and the rest of the trained model is run unchanged against the stored prototypes. A big drop = the decision depends on that branch; a small drop = the other branch can carry it. Zeroing also shifts the fusion layer's input level, so the "average" rows are the fairer measure of information content.</div>
            <table class="eda-table" style="max-width:820px;"><thead><tr><th style="text-align:left;">Variant</th><th>Closed-set accuracy</th><th>Change</th><th>Open-set AUROC</th><th>Change</th></tr></thead><tbody>
            ${ba.variants.map(v => `<tr${v.variant === 'both' ? ' style="font-weight:bold;"' : ''}><td style="text-align:left;">${esc(v.label)}</td><td>${(v.accuracy * 100).toFixed(1)}%</td><td>${v.variant === 'both' ? '' : pts(v.accuracy_drop)}</td><td>${num(v.auroc, 3)}</td><td>${v.variant === 'both' ? '' : auDrop(v.auroc_drop)}</td></tr>`).join('')}
            </tbody></table>
            <div class="rd-kv" style="margin-top:8px; max-width:820px;">
                <div><span>Signal share into fusion layer — 1D-CNN</span><b>${(s.cnn * 100).toFixed(1)}%</b></div>
                <div><span>Signal share into fusion layer — Transformer</span><b>${(s.transformer * 100).toFixed(1)}%</b></div>
                <div><span>Active (non-zero) outputs — 1D-CNN</span><b>${(ba.active_units.cnn * 100).toFixed(0)}% of 64</b></div>
                <div><span>Active (non-zero) outputs — Transformer</span><b>${(ba.active_units.transformer * 100).toFixed(0)}% of 64</b></div>
            </div>
            <div class="hint" style="margin-top:4px;">Signal share = average size of W_h's CNN half × f_cnn vs. its Transformer half × f_tf — how much of the fusion layer's input each branch supplies. Size is not the same as usefulness; the ablation table is the direct test.</div></div>`;
    }

    function renderExplain(body, r, report) {
        const a = report.architecture;
        const fi = report.feature_importance || [];
        const fl = a.first_layer || {};
        body.innerHTML = `
            <div class="hint" style="margin-bottom:12px;">A neural network is not one closed-form equation: it is the chain of layer computations below, with ${a.total_params.toLocaleString()} learned weights.
            Every formula is generated from this trained model's actual architecture and settings. The two "feature influence" measures show which inputs actually drive its decisions.</div>
            <div class="rd-section"><div class="rd-h">How the model computes a decision</div>
                ${a.formulas.map(f => `<div class="rd-formula"><div class="t">${esc(f.title)}</div><pre>${esc(f.lines.join('\n'))}</pre></div>`).join('')}</div>
            <div class="rd-section"><div class="rd-h">Input scaling constants (μ, σ per feature)</div>
                <table class="eda-table" style="max-width:640px;"><thead><tr><th>Feature</th><th>μ (mean)</th><th>σ (std)</th></tr></thead><tbody>
                ${a.scaling.map(s => `<tr><td>${esc(s.label)}</td><td>${num(s.mean)}</td><td>${num(s.std)}</td></tr>`).join('')}
                </tbody></table></div>
            ${fi.length ? `<div class="rd-section"><div class="rd-h">Feature influence</div>
                <div class="hint" style="margin-bottom:6px;"><b>Permutation importance</b>: validation accuracy lost when one feature's values are shuffled across rows (breaking its link to the class) — how much the model relies on it. <b>Gradient sensitivity</b>: average size of the change in the true-class output for a small change in that (scaled) input.</div>
                <div style="height:${Math.max(160, fi.length * 34 + 50)}px;"><canvas id="rd-fi"></canvas></div>
                <table class="eda-table" style="margin-top:8px;"><thead><tr><th>Feature</th><th>Accuracy drop when shuffled</th><th>± std</th><th>Share of total</th><th>Gradient sensitivity</th></tr></thead><tbody>
                ${fi.map(f => `<tr><td>${esc(f.label)}</td><td>${(f.permutation_drop * 100).toFixed(1)} pts</td><td>${(f.permutation_std * 100).toFixed(1)}</td><td>${(f.permutation_share * 100).toFixed(1)}%</td><td>${num(f.gradient_sensitivity)}</td></tr>`).join('')}
                </tbody></table></div>` : ''}
            ${renderBranchAblation(report.branch_ablation, r)}
            <div class="rd-section"><div class="rd-h">Weights by layer (${a.layers.length} tensors, ${a.total_params.toLocaleString()} values)</div>
                <div class="hint" style="margin-bottom:6px;">Summary statistics of every learned weight matrix / bias vector. Download "All Weights (JSON)" above for every individual value.</div>
                <div class="rd-scroll"><table class="eda-table rd-wtable"><thead><tr><th style="text-align:left;">Layer</th><th>Shape</th><th>Values</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>L2 norm</th></tr></thead><tbody>
                ${a.layers.map(l => `<tr><td style="text-align:left;">${esc(l.name)}</td><td>${l.shape.join(' × ')}</td><td>${l.params.toLocaleString()}</td><td>${num(l.mean)}</td><td>${num(l.std)}</td><td>${num(l.min)}</td><td>${num(l.max)}</td><td>${num(l.l2_norm)}</td></tr>`).join('')}
                </tbody></table></div></div>
            ${fl.conv1 ? `<div class="rd-section"><div class="rd-h">First-layer weights in full — 1D-CNN filters (touch the inputs directly)</div>
                <div class="hint" style="margin-bottom:6px;">Each of the 32 filters is 3 weights [w₁ w₂ w₃] applied to every window of 3 neighbouring inputs, plus a bias: h = ReLU(w₁·x̃ⱼ₋₁ + w₂·x̃ⱼ + w₃·x̃ⱼ₊₁ + b).</div>
                <div class="rd-scroll"><table class="eda-table rd-wtable"><thead><tr><th style="text-align:left;">Filter</th><th>w₁</th><th>w₂</th><th>w₃</th><th>bias</th></tr></thead><tbody>
                ${fl.conv1.kernels.map((k, i) => `<tr><td style="text-align:left;">#${i + 1}</td>${k.map(v => `<td>${v.toFixed(4)}</td>`).join('')}<td>${fl.conv1.bias[i].toFixed(4)}</td></tr>`).join('')}
                </tbody></table></div></div>` : ''}
            ${fl.input_proj ? `<div class="rd-section"><div class="rd-h">First-layer weights in full — Transformer input projection</div>
                <div class="hint" style="margin-bottom:6px;">Each scaled input x̃ⱼ becomes a 32-d token tⱼ = x̃ⱼ · w + b, using these 32 weights and biases (shared by all inputs).</div>
                <div class="rd-scroll"><table class="eda-table rd-wtable"><thead><tr><th style="text-align:left;">Dim</th>${fl.input_proj.weights.map((_, i) => `<th>${i + 1}</th>`).join('')}</tr></thead><tbody>
                <tr><td style="text-align:left;">w</td>${fl.input_proj.weights.map(v => `<td>${v.toFixed(3)}</td>`).join('')}</tr>
                <tr><td style="text-align:left;">b</td>${fl.input_proj.bias.map(v => `<td>${v.toFixed(3)}</td>`).join('')}</tr>
                </tbody></table></div></div>` : ''}
            ${fl.feature_embed ? `<div class="rd-section"><div class="rd-h">Per-feature embeddings — Transformer</div>
                <div class="hint" style="margin-bottom:6px;">Each input j also gets its own learned 32-d vector Eⱼ added to its token, so attention knows which feature it is looking at. Size ‖Eⱼ‖ of each (full values in the weights download):</div>
                <div class="rd-scroll"><table class="eda-table rd-wtable"><thead><tr><th style="text-align:left;">Input</th><th>‖Eⱼ‖</th></tr></thead><tbody>
                ${fl.feature_embed.inputs.map((nm, i) => `<tr><td style="text-align:left;">${esc(nm)}</td><td>${fl.feature_embed.norms[i].toFixed(3)}</td></tr>`).join('')}
                </tbody></table></div></div>` : ''}`;
        if (fi.length) {
            chart('rd-fi', {
                type: 'bar',
                data: { labels: fi.map(f => f.label), datasets: [{ label: 'Accuracy drop when shuffled (points)', data: fi.map(f => +(f.permutation_drop * 100).toFixed(2)), backgroundColor: '#00d2ff' }] },
                options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
                    scales: { x: { title: { display: true, text: 'Validation accuracy lost when the feature is shuffled (percentage points)' } } } },
            });
        }
    }

    async function act(url, opts) {
        const resp = await fetch(url, opts);
        const data = await resp.json();
        document.getElementById('rd-msg').innerText = data.message || '';
        if (data.status === 'success') {
            const tab = document.querySelector('.rd-tab.active')?.getAttribute('onclick')?.match(/'(\w+)'/)?.[1] || 'overview';
            await openRunDetails(current.run.id, tab);
            document.getElementById('rd-msg').innerText = data.message || '';
            if (window.onRunDetailsChanged) window.onRunDetailsChanged();
        }
    }

    window.rdSave = async function () {
        const name = prompt('Name for this model (letters, digits, spaces, _ - .):', current.run.saved_name || '');
        if (name === null || !name.trim()) return;
        await act(`/api/runs/${current.run.id}/save`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name.trim() }) });
    };
    window.rdUnsave = async function () {
        if (!confirm(`Unsave '${current.run.saved_name}'? The model stays available while it's among the 20 newest unsaved runs or in use.`)) return;
        await act(`/api/runs/${current.run.id}/save`, { method: 'DELETE' });
    };
    window.rdActivateSeq = async function () {
        await act('/api/sequence/models/activate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ run_id: current.run.id }) });
    };

    // Shared with the Run History table. Returns true when the run was deleted.
    function renderCrossValidation(cv) {
        if (!cv) return '';
        const pct = (x) => x === null || x === undefined ? '—' : (x * 100).toFixed(1) + '%';
        const rows = (cv.per_fold || []).map(f => `<tr${f.fold === cv.best_fold ? ' style="background:rgba(0,210,255,0.08);"' : ''}><td>${f.fold}${f.fold === cv.best_fold ? ' ★' : ''}</td><td>${f.train_tracks}</td><td>${f.validation_tracks}</td><td>${pct(f.closed_set_accuracy)}</td><td>${f.open_set_auroc ?? '—'}</td><td>${f.final_loss}</td></tr>`).join('');
        const t = cv.test_result;
        return `<div class="rd-section"><div class="rd-h">Cross-validation &amp; test split</div>
            <div class="rd-kv">
                <div><span>Test split</span><b>${cv.test_pct}% of the tracks of every class (${(cv.test_instances || []).length} tracks incl. unknown)${cv.include_test ? ' - ALSO trained on, not independent' : ' - never trained on'}</b></div>
                <div><span>Folds</span><b>${cv.folds}${cv.cross_validation ? '' : ' (cross-validation off: fold 1 only)'}</b></div>
                <div><span>Validation accuracy over the folds</span><b>${pct(cv.accuracy_mean)} ± ${pct(cv.accuracy_std)}</b></div>
                <div><span>Model kept</span><b>${esc(cv.kept_model)}</b></div>
                ${t ? `<div><span>Kept model on the test split</span><b>window accuracy ${pct(t.closed_set_accuracy)}, track accuracy ${pct(t.track_accuracy)}, AUROC ${t.open_set_auroc ?? '—'}</b></div>` : ''}
            </div>
            <table class="eda-table" style="margin-top:6px;"><thead><tr><th>Fold</th><th>Train tracks</th><th>Validation tracks</th><th>Window accuracy</th><th>AUROC</th><th>Final loss</th></tr></thead><tbody>${rows}</tbody></table>
            <div class="hint" style="margin-top:4px;">Run the test split again, with any thresholds, on the Testing page (&quot;The model's held-out test split&quot;).</div></div>`;
    }

    window.deleteTrainingRun = async function (runId, savedName) {
        const what = savedName ? `saved model '${savedName}' (run #${runId})` : `run #${runId}`;
        if (!confirm(`Permanently delete ${what}? Its Run History entry, trained model files and its Testing-page results are removed. This can't be undone.`)) return false;
        const resp = await fetch(`/api/runs/${runId}`, { method: 'DELETE' });
        const data = await resp.json();
        alert(data.message);
        return data.status === 'success';
    };
    window.rdDelete = async function () {
        if (!await deleteTrainingRun(current.run.id, current.run.saved_name)) return;
        closeRunDetails();
        if (window.onRunDetailsChanged) window.onRunDetailsChanged();
    };
})();

