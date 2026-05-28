import { useEffect, useRef } from 'react';

/* Animated aurora canvas background -- React Bits style */
export default function Aurora({ className = '' }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let raf;
    let t = 0;

    const resize = () => {
      canvas.width  = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const blobs = [
      { x: 0.3, y: 0.3, r: 0.35, hue: 90,  speed: 0.0008 },
      { x: 0.7, y: 0.5, r: 0.30, hue: 100, speed: 0.0010 },
      { x: 0.5, y: 0.8, r: 0.28, hue: 80,  speed: 0.0006 },
      { x: 0.1, y: 0.6, r: 0.22, hue: 70,  speed: 0.0012 },
    ];

    const draw = () => {
      t += 1;
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      blobs.forEach((b, i) => {
        const bx = (b.x + Math.sin(t * b.speed + i) * 0.15) * w;
        const by = (b.y + Math.cos(t * b.speed * 1.3 + i) * 0.12) * h;
        const r  = b.r * Math.max(w, h);

        const g = ctx.createRadialGradient(bx, by, 0, bx, by, r);
        g.addColorStop(0,   `hsla(${b.hue}, 45%, 35%, 0.22)`);
        g.addColorStop(0.5, `hsla(${b.hue}, 40%, 25%, 0.10)`);
        g.addColorStop(1,   `hsla(${b.hue}, 35%, 15%, 0.00)`);

        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(bx, by, r, 0, Math.PI * 2);
        ctx.fill();
      });

      raf = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 w-full h-full pointer-events-none ${className}`}
    />
  );
}
