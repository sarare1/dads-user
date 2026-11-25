// The one HTML-escape helper for every page: text from the server (dataset, model and column names, run
// details) goes through it before it is put into innerHTML. Load this before any other DADS script.
function escapeHtml(t) {
    return String(t ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}
