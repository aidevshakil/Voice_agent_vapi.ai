import React, { useState, useEffect } from 'react';
import { Upload, FileText, Trash2, RefreshCw, Plus, CheckCircle, AlertCircle, Database } from 'lucide-react';
import { apiFetch } from '../lib/api.js';

export default function Documents() {
  const [docData, setDocData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [pasteText, setPasteText] = useState('');
  const [sourceLabel, setSourceLabel] = useState('pasted-note');
  const [statusMsg, setStatusMsg] = useState(null);

  const fetchDocs = async () => {
    try {
      setLoading(true);
      const res = await apiFetch('/api/v1/documents');
      if (res.ok) {
        const data = await res.json();
        setDocData(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocs();
  }, []);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setStatusMsg(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await apiFetch('/api/v1/documents/upload', {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const result = await res.json();
        setStatusMsg({ type: 'success', text: `Indexed ${result.chunks_indexed} chunk(s) from ${file.name}` });
        fetchDocs();
      } else {
        throw new Error('Upload failed');
      }
    } catch (err) {
      setStatusMsg({ type: 'error', text: err.message });
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const handleTextIndex = async (e) => {
    e.preventDefault();
    if (!pasteText.trim()) return;

    try {
      setUploading(true);
      const res = await apiFetch('/api/v1/documents/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: pasteText, source: sourceLabel }),
      });
      if (res.ok) {
        const result = await res.json();
        setStatusMsg({ type: 'success', text: `Indexed ${result.chunks_indexed} chunk(s)` });
        setPasteText('');
        fetchDocs();
      }
    } catch (err) {
      setStatusMsg({ type: 'error', text: err.message });
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteSource = async (source) => {
    try {
      const res = await apiFetch(`/api/v1/documents/${encodeURIComponent(source)}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        setStatusMsg({ type: 'success', text: `Deleted "${source}"` });
        fetchDocs();
      }
    } catch (err) {
      setStatusMsg({ type: 'error', text: err.message });
    }
  };

  return (
    <div style={{ maxWidth: 1100, margin: '28px auto', padding: '0 24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 800, color: '#fff', marginBottom: 4 }}>
            Knowledge Base
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            {docData ? `${docData.total_vectors} indexed vectors • ${docData.provider}/${docData.collection}` : 'Loading...'}
          </p>
        </div>
      </div>

      {statusMsg && (
        <div style={{
          padding: '10px 16px', borderRadius: 8, marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8,
          background: statusMsg.type === 'success' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)',
          border: `1px solid ${statusMsg.type === 'success' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
          color: statusMsg.type === 'success' ? 'var(--accent-green)' : '#fca5a5', fontSize: '0.88rem'
        }}>
          {statusMsg.type === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
          <span>{statusMsg.text}</span>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 20 }}>
        {/* Document list */}
        <div className="glass-card" style={{ padding: 20 }}>
          <h3 style={{ color: '#fff', fontSize: '1rem', fontWeight: 700, marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Database size={16} color="var(--accent-cyan)" /> Ingested Documents
          </h3>

          {loading ? (
            <div style={{ color: 'var(--text-muted)', padding: 20, textAlign: 'center' }}>Loading...</div>
          ) : docData?.sources?.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {docData.sources.map((source, idx) => (
                <div key={idx} style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '10px 14px', borderRadius: 8, background: 'rgba(255,255,255,0.03)',
                  border: '1px solid var(--border-color)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <FileText size={16} color="var(--accent-cyan)" />
                    <span style={{ fontSize: '0.88rem', fontWeight: 600, color: '#fff' }}>{source}</span>
                  </div>
                  <button className="btn btn-danger" style={{ padding: '4px 10px', fontSize: '0.75rem' }} onClick={() => handleDeleteSource(source)}>
                    <Trash2 size={13} /> Delete
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: 30, color: 'var(--text-muted)', fontSize: '0.88rem' }}>
              No documents indexed yet. Upload a file below.
            </div>
          )}
        </div>

        {/* Upload tool */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="glass-card" style={{ padding: 20, textAlign: 'center' }}>
            <h3 style={{ color: '#fff', fontSize: '0.95rem', fontWeight: 700, marginBottom: 10 }}>Upload File</h3>
            <label style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              padding: '24px 12px', border: '2px dashed var(--border-highlight)', borderRadius: 10,
              background: 'rgba(0, 229, 255, 0.03)', cursor: 'pointer'
            }}>
              <Upload size={28} color="var(--accent-cyan)" style={{ marginBottom: 8 }} />
              <span style={{ fontSize: '0.86rem', fontWeight: 600, color: '#fff' }}>
                {uploading ? 'Processing...' : 'Upload Document'}
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>PDF, TXT, MD, DOCX, CSV</span>
              <input type="file" onChange={handleFileUpload} accept=".pdf,.txt,.md,.docx,.html,.csv,.json" style={{ display: 'none' }} disabled={uploading} />
            </label>
          </div>

          <div className="glass-card" style={{ padding: 20 }}>
            <h3 style={{ color: '#fff', fontSize: '0.95rem', fontWeight: 700, marginBottom: 10 }}>Paste Text Snippet</h3>
            <form onSubmit={handleTextIndex} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <input className="form-input" placeholder="Source label" value={sourceLabel} onChange={(e) => setSourceLabel(e.target.value)} />
              <textarea className="form-textarea" rows={3} placeholder="Paste text..." value={pasteText} onChange={(e) => setPasteText(e.target.value)} />
              <button className="btn btn-primary" type="submit" disabled={uploading || !pasteText.trim()}>
                <Plus size={15} /> Ingest Text
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
