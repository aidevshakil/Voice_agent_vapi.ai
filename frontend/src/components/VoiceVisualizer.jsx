import React, { useEffect, useRef } from 'react';

export default function VoiceVisualizer({ isActive = false, volume = 0 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationId;
    let phase = 0;

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;
      const centerY = height / 2;
      const centerX = width / 2;

      ctx.clearRect(0, 0, width, height);

      const maxRadius = Math.min(width, height) / 3;
      const basePulse = isActive ? 1 + Math.sin(phase * 2) * 0.08 + volume * 0.4 : 1;
      const radius = maxRadius * basePulse;

      const gradient = ctx.createRadialGradient(centerX, centerY, 5, centerX, centerY, radius);
      if (isActive) {
        gradient.addColorStop(0, 'rgba(0, 229, 255, 0.4)');
        gradient.addColorStop(0.5, 'rgba(138, 43, 226, 0.25)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
      } else {
        gradient.addColorStop(0, 'rgba(255, 255, 255, 0.08)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
      }

      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();

      const numLines = 3;
      for (let i = 0; i < numLines; i++) {
        ctx.beginPath();
        ctx.lineWidth = 2.5 - i * 0.5;
        ctx.strokeStyle = isActive 
          ? i === 0 ? '#00e5ff' : i === 1 ? '#8a2be2' : '#ec4899'
          : 'rgba(255, 255, 255, 0.2)';

        for (let x = 0; x < width; x += 4) {
          const normX = (x - centerX) / (width / 2);
          const envelope = Math.exp(-normX * normX * 3);
          const freq = 0.02 + i * 0.01;
          const amp = isActive ? (30 + volume * 80) * envelope : 8 * envelope;
          const y = centerY + Math.sin(x * freq + phase + i * 1.2) * amp;

          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      phase += isActive ? 0.08 : 0.03;
      animationId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationId);
  }, [isActive, volume]);

  return (
    <div style={{ position: 'relative', width: '100%', height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <canvas
        ref={canvasRef}
        width={600}
        height={220}
        style={{ width: '100%', height: '100%', maxWidth: 600, display: 'block' }}
      />
    </div>
  );
}
