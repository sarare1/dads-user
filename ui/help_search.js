/* Help-page search.
 *
 * Core (DOM-free, unit-testable): normalisation, tokenising + light stemming, synonym / acronym
 * expansion, an inverted-style TF-IDF scorer with heading boost, phrase ("quoted") matching,
 * prefix and one-typo matching, AND-first results with a partial-match fallback.
 * UI part (bottom): indexes the page's passages, shows grouped results with highlighted snippets,
 * keyboard navigation ("/" to focus, arrows, Enter, Esc) and jump-to-passage.
 */
(function (root) {
    'use strict';

    // ------------------------------------------------------------------ text normalisation
    const GREEK = { 'τ': ' tau ', 'Δ': ' delta ', 'δ': ' delta ', 'λ': ' lambda ', 'σ': ' sigma ', 'μ': ' mu ', 'ε': ' epsilon ',
                    'α': ' alpha ', 'β': ' beta ', 'γ': ' gamma ', 'θ': ' theta ', 'π': ' pi ', 'ω': ' omega ', 'Σ': ' sum ', '≥': ' ', '≤': ' ' };
    const STOP = new Set(['the', 'a', 'an', 'of', 'to', 'is', 'in', 'on', 'and', 'or', 'for', 'it', 'as', 'by', 'at', 'be', 'this', 'that',
                          'with', 'from', 'are', 'was', 'what', 'how', 'do', 'does', 'i', 'my', 'can', 'if', 'so', 'than', 'then', 'its']);

    function normalize(text) {
        return String(text || '').replace(/[τΔδλσμεαβγθπωΣ≥≤]/g, ch => GREEK[ch] || ' ')
            .replace(/([a-z])([A-Z])/g, '$1 $2')      // camelCase → camel Case
            .toLowerCase()
            .replace(/[_\-–—/\\.,;:()[\]{}"'`“”‘’|=+*<>^~!?%]/g, ' ')
            .replace(/\s+/g, ' ').trim();
    }

    function stem(w) {
        if (w.length <= 3) return w;
        if (w.endsWith('ies') && w.length > 4) return w.slice(0, -3) + 'y';
        if (w.endsWith('sses')) return w.slice(0, -2);
        if (w.endsWith('ing') && w.length > 5) return w.slice(0, -3);
        if (w.endsWith('ed') && w.length > 4) return w.slice(0, -2);
        if (w.endsWith('es') && w.length > 4) return w.slice(0, -2);
        if (w.endsWith('s') && !w.endsWith('ss') && w.length > 3) return w.slice(0, -1);
        return w;
    }

    function tokens(text, keepStop) {
        return normalize(text).split(' ').filter(t => t && (keepStop || !STOP.has(t)) && (t.length > 1 || /\d/.test(t))).map(stem);
    }

    // ------------------------------------------------------------------ synonyms / acronyms (each row = one concept)
    const SYNONYMS = [
        ['threshold', 'tau', 'cutoff', 'limit', 'calibration', 'calibrate'],
        ['unknown', 'ood', 'anomaly', 'novel', 'holdout', 'out of distribution', 'open set', 'openset'],
        ['stop', 'cancel', 'abort', 'interrupt'],
        ['emitter', 'radar', 'family', 'class', 'source'],
        ['accuracy', 'correct', 'score', 'performance', 'metric'],
        ['auroc', 'roc', 'auc'],
        ['pri', 'pulse repetition interval', 'prf', 'interval'],
        ['frequency', 'freq', 'carrier', 'fc'],
        ['pw', 'pulse width', 'width'],
        ['upload', 'import', 'csv', 'custom', 'own data'],
        ['train', 'training', 'fit', 'learn', 'epoch'],
        ['incremental', 'new family', 'add family', 'lwf', 'forgetting', 'continual'],
        ['noise', 'spurious', 'clutter'],
        ['window', 'k', 'segment'],
        ['prototype', 'centroid', 'kmeans', 'k means', 'cluster'],
        ['test', 'testing', 'evaluate', 'evaluation', 'validation'],
        ['inference', 'classify', 'predict', 'infer'],
        ['ecm', 'countermeasure', 'jammer', 'jamming'],
        ['confusion', 'matrix', 'mistake', 'misclassified'],
        ['feature', 'input', 'column', 'attribute'],
        ['embedding', 'embed', 'vector', 'latent'],
        ['hyperparameter', 'hyperparam', 'setting', 'config', 'configuration', 'learning rate'],
        ['model', 'network', 'backbone', 'architecture'],
        ['dataset', 'data set', 'tracks', 'data'],
        ['precision', 'recall', 'f1'],
        ['gru', 'rnn', 'recurrent'], ['cnn', 'convolution', 'convolutional'], ['transformer', 'attention'],
        ['verified', 'known', 'accepted'],
        ['delta', 'change', 'difference'],
        ['cross validation', 'cv', 'fold', 'kfold'],
    ];
    const SYN_INDEX = (() => {
        const idx = new Map();
        SYNONYMS.forEach((row, r) => row.forEach(term => {
            const key = tokens(term, true).join(' ');
            if (!idx.has(key)) idx.set(key, new Set());
            idx.get(key).add(r);
        }));
        return idx;
    })();

    function expansions(term) {
        const rows = SYN_INDEX.get(term);
        if (!rows) return [];
        const out = new Set();
        rows.forEach(r => SYNONYMS[r].forEach(s => tokens(s, true).forEach(t => { if (t !== term) out.add(t); })));
        return [...out];
    }

    // ------------------------------------------------------------------ fuzzy
    function withinOneEdit(a, b) {   // one insertion / deletion / substitution / adjacent swap
        if (Math.abs(a.length - b.length) > 1) return false;
        if (a.length === b.length) {
            let i = 0;
            while (i < a.length && a[i] === b[i]) i++;
            if (i === a.length) return true;
            if (a.slice(i + 1) === b.slice(i + 1)) return true;                                  // substitution
            return a[i] === b[i + 1] && a[i + 1] === b[i] && a.slice(i + 2) === b.slice(i + 2);   // swap
        }
        const [long, short] = a.length > b.length ? [a, b] : [b, a];
        let i = 0;
        while (i < short.length && long[i] === short[i]) i++;
        return long.slice(i + 1) === short.slice(i);                                              // insertion / deletion
    }

    // ------------------------------------------------------------------ index + search
    // passage = { id, heading, section, text }
    function buildIndex(passages) {
        const docs = passages.map(p => {
            const body = tokens(p.text), head = tokens(p.heading + ' ' + (p.section || ''));
            const tf = new Map(), hf = new Set(head);
            body.forEach(t => tf.set(t, (tf.get(t) || 0) + 1));
            return { p, tf, hf, norm: ' ' + normalize(p.text + ' ' + p.heading) + ' ', len: body.length };
        });
        const df = new Map();
        docs.forEach(d => new Set([...d.tf.keys(), ...d.hf]).forEach(t => df.set(t, (df.get(t) || 0) + 1)));
        const vocab = [...df.keys()];
        return { docs, df, vocab, N: docs.length };
    }

    const idf = (ix, t) => Math.log(1 + ix.N / (1 + (ix.df.get(t) || 0)));

    // The vocabulary tokens a query term stands for — itself (1), tokens it is a prefix of (0.6), tokens one
    // typo away (0.4) — worked out once per query, then looked up per passage.
    function termTokens(ix, term, w) {
        const out = new Map();
        const add = (t, weight) => { if ((out.get(t) || 0) < weight) out.set(t, weight); };
        if (ix.df.has(term)) add(term, w);
        if (term.length >= 3) for (const t of ix.vocab) if (t !== term && t.startsWith(term)) add(t, w * 0.6);
        if (!out.size && term.length >= 5) for (const t of ix.vocab) if (withinOneEdit(term, t)) add(t, w * 0.4);
        return out;
    }

    // how well one passage matches a term (with its precomputed tokens): 0 = no match
    function matchTerm(ix, d, tokenWeights) {
        let best = 0;
        tokenWeights.forEach((weight, t) => {
            const f = d.tf.get(t) || 0, inHead = d.hf.has(t);
            if (!f && !inHead) return;
            best = Math.max(best, weight * idf(ix, t) * ((f ? 1 + Math.log(f) : 0) + (inHead ? 4 : 0)));
        });
        return best;
    }

    function parseQuery(q) {
        const phrases = [];
        const rest = String(q || '').replace(/"([^"]+)"/g, (_, ph) => { phrases.push(normalize(ph)); return ' '; });
        const terms = [...new Set(tokens(rest))];
        return { terms, phrases: phrases.filter(Boolean) };
    }

    function search(ix, query, limit) {
        const { terms, phrases } = parseQuery(query);
        if (!terms.length && !phrases.length) return { results: [], partial: false, terms: [] };
        const allTerms = terms.slice();
        phrases.forEach(ph => tokens(ph).forEach(t => { if (!allTerms.includes(t)) allTerms.push(t); }));
        const scored = [];
        const direct = allTerms.map(t => termTokens(ix, t, 1));
        const viaSynonym = allTerms.map(t => {
            const m = new Map();
            expansions(t).forEach(e => termTokens(ix, e, 0.45).forEach((w, tok) => { if ((m.get(tok) || 0) < w) m.set(tok, w); }));
            return m;
        });
        for (const d of ix.docs) {
            if (phrases.some(ph => !d.norm.includes(' ' + ph + ' ') && !d.norm.includes(ph))) continue;
            let score = 0, matched = 0;
            for (let i = 0; i < allTerms.length; i++) {
                let s = matchTerm(ix, d, direct[i]);
                if (!s) s = matchTerm(ix, d, viaSynonym[i]);
                if (s) { matched++; score += s; }
            }
            if (!matched) continue;
            const full = matched === allTerms.length;
            if (terms.length > 1) {
                const joined = terms.join(' ');
                if (d.norm.includes(' ' + normalize(query.replace(/"/g, '')) + ' ')) score += 8;   // the words together, in order
                else if (tokens(d.p.text).join(' ').includes(joined)) score += 5;
            }
            phrases.forEach(() => { score += 8; });
            if (full) score *= 1.6;
            scored.push({ p: d.p, score, full, matched });
        }
        const complete = scored.filter(s => s.full);
        const pool = complete.length ? complete : scored;
        pool.sort((a, b) => b.score - a.score);
        return { results: pool.slice(0, limit || 60), partial: !complete.length && scored.length > 0, terms: allTerms };
    }

    const core = { normalize, stem, tokens, expansions, buildIndex, search, parseQuery, withinOneEdit };
    if (typeof module !== 'undefined' && module.exports) module.exports = core;
    root.HelpSearchCore = core;

    // ================================================================== UI
    if (typeof document === 'undefined') return;

    const esc = escapeHtml;  // ui/escape.js
    let index = null, passageEls = [], activeResults = [], activePos = -1, debounce = null;

    function textOf(node) {
        if (node.nodeType === 3) return node.nodeValue;
        if (node.nodeType !== 1 || /^(SCRIPT|STYLE|CANVAS)$/.test(node.tagName)) return '';
        let out = '';
        node.childNodes.forEach(c => { out += (c.nodeType === 1 ? ' ' : '') + textOf(c) + (c.nodeType === 1 ? ' ' : ''); });
        return out;
    }

    function buildPageIndex() {
        const container = document.querySelector('.page-content');
        const sel = 'h2,h3,p,li,tr,.glossary-term,.glossary-def,.definition,.hint,.why,.formula,.twocol > div,.dgrid > *,pre';
        let els = [...container.querySelectorAll(sel)].filter(e => !e.closest('#help-search') && !e.closest('.toc'));
        // keep the innermost passages: a block that contains another candidate is represented by it
        els = els.filter(e => !/^H[23]$/.test(e.tagName) ? !e.querySelector(sel) : true);
        let h2 = '', h3 = '', term = '';
        const passages = [];
        passageEls = [];
        els.forEach(e => {
            const text = textOf(e).replace(/\s+/g, ' ').trim();
            if (!text) return;
            if (e.tagName === 'H2') { h2 = text; h3 = ''; }
            else if (e.tagName === 'H3') h3 = text;
            if (e.classList.contains('glossary-term')) { term = text; return; }
            const heading = e.tagName === 'H2' || e.tagName === 'H3' ? text : (e.classList.contains('glossary-def') ? term : (e.tagName === 'TR' ? text.split(' ')[0] : ''));
            const section = [h2, h3].filter(Boolean).join(' › ');
            const full = e.classList.contains('glossary-def') ? `${term}. ${text}` : text;
            passages.push({ id: passages.length, heading, section: h3 || h2, crumb: section || 'Help', text: full });
            passageEls.push(e);
        });
        index = HelpSearchCore.buildIndex(passages);
    }

    function snippet(text, terms) {
        const lower = HelpSearchCore.normalize(text);
        const words = text.split(/(\s+)/);
        const stems = terms.slice();
        const hit = (w) => { const n = HelpSearchCore.tokens(w)[0]; return n && stems.some(t => n === t || (t.length >= 3 && n.startsWith(t))); };
        let first = words.findIndex(hit);
        if (first < 0) first = 0;
        const from = Math.max(0, first - 12), to = Math.min(words.length, first + 34);
        const html = words.slice(from, to).map(w => /^\s+$/.test(w) ? w : (hit(w) ? `<mark>${esc(w)}</mark>` : esc(w))).join('');
        return (from > 0 ? '… ' : '') + html + (to < words.length ? ' …' : '');
    }

    function render(box, q) {
        const out = HelpSearchCore.search(index, q, 40);
        activeResults = out.results; activePos = -1;
        if (!out.results.length) {
            box.innerHTML = `<div class="hs-empty">No match for “${esc(q)}”. Try fewer or different words — synonyms work too (e.g. <i>threshold</i>, <i>tau</i>, <i>unknown</i>, <i>stop</i>).</div>`;
            return;
        }
        const groups = new Map();
        out.results.forEach((r, i) => { const k = r.p.crumb; if (!groups.has(k)) groups.set(k, []); groups.get(k).push([r, i]); });
        let html = `<div class="hs-count">${out.results.length} result${out.results.length > 1 ? 's' : ''} in ${groups.size} section${groups.size > 1 ? 's' : ''}${out.partial ? ' — no passage has all the words, showing partial matches' : ''}</div>`;
        groups.forEach((items, crumb) => {
            html += `<div class="hs-group"><div class="hs-crumb">${esc(crumb)}</div>` + items.slice(0, 4).map(([r, i]) =>
                `<div class="hs-item" data-i="${i}">${r.p.heading && !r.p.text.startsWith(r.p.heading) ? `<b>${esc(r.p.heading)}</b> ` : ''}${snippet(r.p.text, out.terms)}</div>`).join('') +
                (items.length > 4 ? `<div class="hs-more">+${items.length - 4} more in this section</div>` : '') + '</div>';
        });
        box.innerHTML = html;
    }

    function go(i) {
        const r = activeResults[i];
        if (!r) return;
        const el = passageEls[r.p.id];
        let d = el.closest('details');
        while (d) { d.open = true; d = d.parentElement && d.parentElement.closest('details'); }
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        el.classList.remove('hs-flash'); void el.offsetWidth; el.classList.add('hs-flash');
        document.getElementById('help-search-results').style.display = 'none';
    }

    function highlightActive(box) {
        box.querySelectorAll('.hs-item').forEach(e => e.classList.toggle('active', Number(e.dataset.i) === activePos));
        const cur = box.querySelector('.hs-item.active');
        if (cur) cur.scrollIntoView({ block: 'nearest' });
    }

    function init() {
        const input = document.getElementById('help-search-input');
        const box = document.getElementById('help-search-results');
        if (!input || !box) return;
        const run = () => {
            const q = input.value.trim();
            if (!q) { box.style.display = 'none'; return; }
            if (!index) buildPageIndex();
            render(box, q);
            box.style.display = '';
        };
        input.addEventListener('input', () => { clearTimeout(debounce); debounce = setTimeout(run, 120); });
        input.addEventListener('focus', () => { if (input.value.trim()) run(); });
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') { input.value = ''; box.style.display = 'none'; input.blur(); }
            else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                const n = box.querySelectorAll('.hs-item').length;
                if (!n) return;
                const order = [...box.querySelectorAll('.hs-item')].map(x => Number(x.dataset.i));
                const cur = order.indexOf(activePos);
                activePos = order[(cur + (e.key === 'ArrowDown' ? 1 : -1) + n) % n];
                highlightActive(box);
            } else if (e.key === 'Enter') { go(activePos >= 0 ? activePos : (box.querySelector('.hs-item') ? Number(box.querySelector('.hs-item').dataset.i) : -1)); }
        });
        box.addEventListener('mousedown', (e) => { const it = e.target.closest('.hs-item'); if (it) { e.preventDefault(); go(Number(it.dataset.i)); } });
        document.addEventListener('click', (e) => { if (!e.target.closest('#help-search')) box.style.display = 'none'; });
        document.addEventListener('keydown', (e) => {
            if (e.key === '/' && !/^(INPUT|TEXTAREA|SELECT)$/.test((document.activeElement || {}).tagName)) { e.preventDefault(); input.focus(); input.select(); }
        });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})(typeof window !== 'undefined' ? window : globalThis);
