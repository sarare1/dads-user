// Sequence model selector (Inference page, Live Simulation page).
// Choosing a model makes it the sequence model in use server-wide (Inference, Live Simulation).
//
//   renderSequenceSelector(containerId, { onSwitched(model) })  → Promise<active model or null>

async function renderSequenceSelector(containerId, opts = {}) {
    const el = document.getElementById(containerId);
    const esc = escapeHtml;  // ui/escape.js
    const data = await (await fetch('/api/sequence/models')).json();
    const models = data.models || [];
    if (!models.length) {
        el.innerHTML = `<div class="hint">No pulse-sequence model yet — train one on the <a href="/" style="color:var(--accent-blue);">Training page</a> (Sequence Model card).</div>`;
        return null;
    }
    const label = (m) => `${m.name ? '★ ' + m.name + ' — ' : ''}run #${m.run_id} (${m.backbone_type}, k=${m.window_k}, acc ${(m.closed_set_accuracy * 100).toFixed(1)}%${m.dataset_name ? ', dataset ' + m.dataset_name : ''}${m.parent_run_id ? ', ← #' + m.parent_run_id + ' ' + m.incremental_method : ''})`;
    el.innerHTML = `<label style="margin-top:0;">Sequence Model</label>
        <select class="seq-model-select">${models.map(m => `<option value="${m.run_id}" ${m.is_active ? 'selected' : ''}>${esc(label(m))}</option>`).join('')}</select>
        <div class="hint seq-model-hint"></div>`;
    const sel = el.querySelector('.seq-model-select');
    const hint = el.querySelector('.seq-model-hint');
    const describe = (m) => { hint.innerText = `Trained on dataset: ${m.dataset_name || 'unnamed (seed ' + (m.dataset_seed ?? '?') + ')'}. Thresholds: τ_c = ${m.tau_c}, τ_d = ${m.tau_d}. Features: ${(m.features || '').split(',').join(', ')}`; };
    let active = models.find(m => m.is_active) || models[0];
    describe(active);
    sel.onchange = async () => {
        const resp = await (await fetch('/api/sequence/models/activate', { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ run_id: parseInt(sel.value) }) })).json();
        if (resp.status !== 'success') { alert(resp.message); sel.value = String(active.run_id); return; }
        active = models.find(m => m.run_id === parseInt(sel.value));
        describe(active);
        if (opts.onSwitched) opts.onSwitched(active);
    };
    return active;
}
