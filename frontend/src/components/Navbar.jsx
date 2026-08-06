import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { Mic, Database, Cpu } from 'lucide-react';
import { apiFetch } from '../lib/api.js';

export default function Navbar() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await apiFetch('/api/v1/health/ready');
        if (res.ok) {
          const data = await res.json();
          setHealth(data);
        }
      } catch (err) {
        console.error('Failed to fetch health status', err);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const storeInfo = health?.components?.vector_store || {};
  const llmInfo = health?.components?.llm || {};
  const isReady = health?.status === 'ok';

  return (
    <nav style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '14px 28px',
      borderBottom: '1px solid var(--border-color)',
      background: 'rgba(11, 14, 20, 0.9)',
      backdropFilter: 'blur(16px)',
      position: 'sticky',
      top: 0,
      zIndex: 100
    }}>
      {/* Brand */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{
          width: 36,
          height: 36,
          borderRadius: 10,
          background: 'var(--gradient-btn)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 15px rgba(0, 229, 255, 0.4)'
        }}>
          <Mic size={18} color="#fff" />
        </div>
        <div>
          <div style={{ fontWeight: 800, fontSize: '1.1rem', color: '#fff' }}>
            ⟡ Voice<span style={{ color: 'var(--accent-cyan)' }}>RAG</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            Powered by Vapi.ai
          </div>
        </div>
      </div>

      {/* Nav Links */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 4,
        background: 'rgba(255, 255, 255, 0.03)',
        border: '1px solid var(--border-color)',
        borderRadius: 12,
        padding: 4
      }}>
        <NavLink
          to="/"
          end
          style={({ isActive }) => ({
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '8px 18px',
            borderRadius: 8,
            fontSize: '0.88rem',
            fontWeight: isActive ? 600 : 500,
            color: isActive ? '#ffffff' : 'var(--text-muted)',
            background: isActive ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
            textDecoration: 'none',
            transition: 'all 0.15s ease'
          })}
        >
          <Mic size={16} /> Assistant
        </NavLink>

        <NavLink
          to="/documents"
          style={({ isActive }) => ({
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '8px 18px',
            borderRadius: 8,
            fontSize: '0.88rem',
            fontWeight: isActive ? 600 : 500,
            color: isActive ? '#ffffff' : 'var(--text-muted)',
            background: isActive ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
            textDecoration: 'none',
            transition: 'all 0.15s ease'
          })}
        >
          <Database size={16} /> Knowledge Base
        </NavLink>
      </div>

      {/* Health Badge */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div className={`status-badge ${isReady ? '' : 'warning'}`}>
          <span className="status-dot"></span>
          {isReady ? 'Ready' : 'Degraded'}
        </div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
          <Cpu size={14} color="var(--accent-cyan)" />
          <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{storeInfo.vectors || 0} chunks</span>
        </div>
      </div>
    </nav>
  );
}
