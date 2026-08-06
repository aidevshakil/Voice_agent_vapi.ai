// Where the FastAPI backend lives.
//
// Empty in development: Vite's dev proxy forwards /api to localhost:8000, so
// requests stay same-origin. In a Vercel build, set VITE_API_BASE to the public
// backend URL (e.g. https://voice-rag-api.onrender.com) -- the value is baked in
// at build time, so changing it means redeploying, and the backend's
// APP_CORS_ORIGINS has to name the Vercel domain or the browser blocks the call.
const BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '');

/** Absolute URL for an API path. Pass paths starting with "/api/...". */
export function apiUrl(path) {
  return `${BASE}${path}`;
}

/** fetch() against the configured backend. Same signature otherwise. */
export function apiFetch(path, options) {
  return fetch(apiUrl(path), options);
}
