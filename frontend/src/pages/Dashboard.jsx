import React, { useState, useRef, useEffect } from 'react';
import { PhoneCall, PhoneOff, Mic, MicOff, Send, Bot, User, BookOpen, Trash2, ChevronDown, ChevronRight, Sliders } from 'lucide-react';
import Vapi from '@vapi-ai/web';
import VoiceVisualizer from '../components/VoiceVisualizer.jsx';

const DEFAULT_PUBLIC_KEY = '4be20fdb-ec75-458a-9a3d-d3128f76d54b';
const DEFAULT_ASSISTANT_ID = '20e76e61-bbee-49e6-ab91-472cad59c30f';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('chat');
  
  const [vapi, setVapi] = useState(null);
  const [vapiConfig, setVapiConfig] = useState({
    publicKey: DEFAULT_PUBLIC_KEY,
    assistantId: DEFAULT_ASSISTANT_ID,
  });
  const [callStatus, setCallStatus] = useState('idle');
  const [isMuted, setIsMuted] = useState(false);
  const [volumeLevel, setVolumeLevel] = useState(0);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [openCitations, setOpenCitations] = useState({});

  const [topK, setTopK] = useState(4);
  const [mode, setMode] = useState('text');
  const [useCache, setUseCache] = useState(true);

  const chatEndRef = useRef(null);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const res = await fetch('/api/v1/health/ready');
        const data = await res.json();
        const vapiInfo = data?.components?.vapi || {};
        setVapiConfig({
          publicKey: vapiInfo.public_key || DEFAULT_PUBLIC_KEY,
          assistantId: vapiInfo.assistant_id || DEFAULT_ASSISTANT_ID,
        });
      } catch (e) {
        setVapiConfig({
          publicKey: DEFAULT_PUBLIC_KEY,
          assistantId: DEFAULT_ASSISTANT_ID,
        });
      }
    };
    fetchConfig();
  }, []);

  useEffect(() => {
    const key = vapiConfig.publicKey || DEFAULT_PUBLIC_KEY;
    if (!key) return;

    try {
      const vapiInstance = new Vapi(key);

      vapiInstance.on('call-start', () => {
        setCallStatus('active');
      });

      vapiInstance.on('call-end', () => {
        setCallStatus('idle');
        setVolumeLevel(0);
        setIsMuted(false);
      });

      vapiInstance.on('volume-level', (vol) => {
        setVolumeLevel(vol);
      });

      vapiInstance.on('error', (err) => {
        console.error('Vapi SDK error:', err);
        setCallStatus('idle');
      });

      setVapi(vapiInstance);

      return () => {
        vapiInstance.stop();
      };
    } catch (err) {
      console.error('Failed to initialize Vapi instance:', err);
    }
  }, [vapiConfig.publicKey]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isStreaming]);

  const handleStartCall = () => {
    const key = vapiConfig.publicKey || DEFAULT_PUBLIC_KEY;
    const astId = vapiConfig.assistantId || DEFAULT_ASSISTANT_ID;

    setCallStatus('loading');

    if (vapi) {
      vapi.start(astId);
    } else {
      try {
        const vapiInstance = new Vapi(key);
        vapiInstance.on('call-start', () => setCallStatus('active'));
        vapiInstance.on('call-end', () => {
          setCallStatus('idle');
          setVolumeLevel(0);
          setIsMuted(false);
        });
        vapiInstance.on('volume-level', (vol) => setVolumeLevel(vol));
        vapiInstance.on('error', () => setCallStatus('idle'));
        setVapi(vapiInstance);
        vapiInstance.start(astId);
      } catch (err) {
        setCallStatus('idle');
      }
    }
  };

  const handleEndCall = () => {
    if (vapi) {
      vapi.stop();
    }
    setCallStatus('idle');
  };

  const handleToggleMute = () => {
    if (vapi) {
      const newMuteState = !isMuted;
      vapi.setMuted(newMuteState);
      setIsMuted(newMuteState);
    }
  };

  const handleSend = async (e) => {
    e?.preventDefault();
    if (!input.trim() || isStreaming) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    const prompt = input;
    setInput('');
    setIsStreaming(true);

    setMessages((prev) => [...prev, { role: 'assistant', content: '', citations: [] }]);

    try {
      const response = await fetch('/api/v1/rag/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: prompt,
          session_id: sessionId,
          mode: mode,
          top_k: topK,
          use_cache: useCache,
        }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullText = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6).trim();
            if (!jsonStr || jsonStr === '{}') continue;
            try {
              const data = JSON.parse(jsonStr);
              if (data.session_id) setSessionId(data.session_id);
              if (data.delta) {
                fullText += data.delta;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1].content = fullText;
                  return updated;
                });
              }
            } catch (err) {}
          }
        }
      }

      try {
        const citRes = await fetch('/api/v1/rag/retrieve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: prompt, top_k: topK, use_cache: useCache }),
        });
        if (citRes.ok) {
          const citData = await citRes.json();
          const citations = (citData.chunks || []).map((chunk) => ({
            source: chunk.source,
            score: chunk.score,
            page: chunk.metadata?.page,
            snippet: chunk.text.slice(0, 350),
          }));
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1].citations = citations;
            return updated;
          });
        }
      } catch (e) {}
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1].content = `Request failed: ${err.message}`;
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div style={{ display: 'flex', gap: 20, padding: 24, height: 'calc(100vh - 65px)', maxWidth: 1400, margin: '0 auto' }}>
      <div className="glass-card" style={{ width: 340, display: 'flex', flexDirection: 'column', padding: 24, flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
          <div style={{
            width: 10,
            height: 10,
            borderRadius: '50%',
            background: callStatus === 'active' ? 'var(--accent-green)' : callStatus === 'loading' ? 'var(--accent-amber)' : 'var(--accent-cyan)'
          }} />
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>Voice Assistant</h2>
        </div>

        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <VoiceVisualizer isActive={callStatus === 'active'} volume={volumeLevel} />
        </div>

        <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
          {callStatus === 'idle' ? (
            <button className="btn btn-primary" onClick={handleStartCall} style={{ width: '100%', padding: '12px 20px', borderRadius: 24 }}>
              <PhoneCall size={18} /> Start Voice Call
            </button>
          ) : callStatus === 'loading' ? (
            <button className="btn btn-secondary" disabled style={{ width: '100%', padding: '12px 20px', borderRadius: 24 }}>
              Connecting to Mic...
            </button>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <button className={`btn ${isMuted ? 'btn-secondary' : 'btn-primary'}`} onClick={handleToggleMute}>
                {isMuted ? <MicOff size={16} /> : <Mic size={16} />}
                {isMuted ? 'Muted' : 'Mute'}
              </button>
              <button className="btn btn-danger" onClick={handleEndCall}>
                <PhoneOff size={16} /> End Call
              </button>
            </div>
          )}
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: 4 }}>
            {callStatus === 'active' ? '🎙️ Live voice call connected' : 'Sub-second real-time grounded speech turn-taking'}
          </p>
        </div>
      </div>

      <div className="glass-card" style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '12px 20px', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(255,255,255,0.02)' }}>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={() => setActiveTab('chat')}
              style={{
                padding: '7px 16px', borderRadius: 8, fontSize: '0.86rem', fontWeight: 600,
                background: activeTab === 'chat' ? 'rgba(255,255,255,0.1)' : 'transparent',
                color: activeTab === 'chat' ? '#fff' : 'var(--text-muted)',
                border: '1px solid var(--border-color)', cursor: 'pointer'
              }}
            >
              💬 Text Chat
            </button>
            <button
              onClick={() => setActiveTab('sources')}
              style={{
                padding: '7px 16px', borderRadius: 8, fontSize: '0.86rem', fontWeight: 600,
                background: activeTab === 'sources' ? 'rgba(255,255,255,0.1)' : 'transparent',
                color: activeTab === 'sources' ? '#fff' : 'var(--text-muted)',
                border: '1px solid var(--border-color)', cursor: 'pointer'
              }}
            >
              📚 Sources & Inspector
            </button>
          </div>

          {messages.length > 0 && activeTab === 'chat' && (
            <button className="btn btn-secondary" onClick={() => setMessages([])} style={{ padding: '4px 10px', fontSize: '0.78rem' }}>
              <Trash2 size={13} /> Clear
            </button>
          )}
        </div>

        <div style={{ display: activeTab === 'chat' ? 'flex' : 'none', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
          <div style={{ flex: 1, padding: 20, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 14 }}>
            {messages.length === 0 ? (
              <div style={{ margin: 'auto', textAlign: 'center', color: 'var(--text-muted)', maxWidth: 360 }}>
                <Bot size={36} color="var(--accent-cyan)" style={{ marginBottom: 10 }} />
                <h3 style={{ color: '#fff', fontSize: '1rem', marginBottom: 4 }}>Real-Time Document Assistant</h3>
                <p style={{ fontSize: '0.84rem' }}>Ask any question about your documents. Answers stream sentence-by-sentence with source citations.</p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div key={idx} style={{ display: 'flex', gap: 12, alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '85%' }}>
                  {msg.role === 'assistant' && (
                    <div style={{ width: 30, height: 30, borderRadius: 8, background: 'var(--gradient-btn)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                      <Bot size={16} color="#fff" />
                    </div>
                  )}
                  <div>
                    <div style={{
                      padding: '10px 16px', borderRadius: msg.role === 'user' ? '14px 14px 2px 14px' : '14px 14px 14px 2px',
                      background: msg.role === 'user' ? 'rgba(0, 229, 255, 0.15)' : 'rgba(255, 255, 255, 0.05)',
                      border: `1px solid ${msg.role === 'user' ? 'rgba(0, 229, 255, 0.3)' : 'var(--border-color)'}`,
                      color: '#fff', fontSize: '0.92rem', lineHeight: 1.5, whiteSpace: 'pre-wrap'
                    }}>
                      {msg.content || (isStreaming && idx === messages.length - 1 ? 'Searching vector store...' : '')}
                    </div>

                    {msg.citations?.length > 0 && (
                      <div style={{ marginTop: 6 }}>
                        <button
                          onClick={() => setOpenCitations((p) => ({ ...p, [idx]: !p[idx] }))}
                          style={{ background: 'none', border: 'none', color: 'var(--accent-cyan)', fontSize: '0.78rem', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
                        >
                          {openCitations[idx] ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                          <BookOpen size={12} /> {msg.citations.length} Grounded Source(s)
                        </button>
                        {openCitations[idx] && (
                          <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 6 }}>
                            {msg.citations.map((c, cIdx) => (
                              <div key={cIdx} className="glass-card" style={{ padding: '8px 12px', fontSize: '0.8rem', background: 'rgba(0,0,0,0.3)' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                                  <strong style={{ color: 'var(--accent-cyan)' }}>📄 {c.source} {c.page ? `· p.${c.page}` : ''}</strong>
                                  <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>{(c.score * 100).toFixed(0)}% score</span>
                                </div>
                                <p style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.76rem' }}>"{c.snippet}"</p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  {msg.role === 'user' && (
                    <div style={{ width: 30, height: 30, borderRadius: 8, background: 'rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                      <User size={16} color="#fff" />
                    </div>
                  )}
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>

          <form onSubmit={handleSend} style={{ padding: 14, borderTop: '1px solid var(--border-color)', display: 'flex', gap: 10, background: 'rgba(11,14,20,0.6)' }}>
            <input
              className="form-input"
              placeholder="Ask a question about your indexed documents..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isStreaming}
            />
            <button className="btn btn-primary" type="submit" disabled={isStreaming || !input.trim()}>
              <Send size={16} />
            </button>
          </form>
        </div>

        <div style={{ display: activeTab === 'sources' ? 'flex' : 'none', flexDirection: 'column', flex: 1, padding: 24, overflowY: 'auto' }}>
          <h3 style={{ color: '#fff', fontSize: '1rem', fontWeight: 700, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Sliders size={16} color="var(--accent-cyan)" /> Retrieval Tuning & Controls
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
            <div className="glass-card" style={{ padding: 16 }}>
              <label style={{ fontSize: '0.84rem', color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
                Top K Chunks: <strong style={{ color: '#fff' }}>{topK}</strong>
              </label>
              <input type="range" min="1" max="12" value={topK} onChange={(e) => setTopK(Number(e.target.value))} style={{ width: '100%', accentColor: 'var(--accent-cyan)' }} />
            </div>

            <div className="glass-card" style={{ padding: 16 }}>
              <label style={{ fontSize: '0.84rem', color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>Answer Mode</label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <button type="button" className={`btn ${mode === 'text' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setMode('text')} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>Text</button>
                <button type="button" className={`btn ${mode === 'voice' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setMode('voice')} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>Voice Spoken</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
