import { useEffect, useRef } from 'react';

/* Animated aurora canvas background — Light theme adjusted */
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

    // Lighter, more subtle blobs for the light theme
    const blobs = [
      { x: 0.25, y: 0.25, r: 0.38, color: [14,116,144],  speed: 0.0007 },  // teal
      { x: 0.75, y: 0.45, r: 0.32, color: [197,165,90],   speed: 0.0009 },  // gold
      { x: 0.50, y: 0.80, r: 0.30, color: [8,145,178],    speed: 0.0005 },  // teal-light
      { x: 0.10, y: 0.65, r: 0.25, color: [14,116,144],   speed: 0.0011 },  // teal
      { x: 0.85, y: 0.20, r: 0.22, color: [197,165,90],   speed: 0.0008 },  // gold
    ];

    const draw = () => {
      t += 1;
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      blobs.forEach((b, i) => {
        const bx = (b.x + Math.sin(t * b.speed + i) * 0.18) * w;
        const by = (b.y + Math.cos(t * b.speed * 1.3 + i) * 0.14) * h;
        const r  = b.r * Math.max(w, h);
        const [cr,cg,cb] = b.color;

        const g = ctx.createRadialGradient(bx, by, 0, bx, by, r);
        // Significantly reduced opacity for a soft light theme effect
        g.addColorStop(0,   `rgba(${cr},${cg},${cb}, 0.06)`);
        g.addColorStop(0.4, `rgba(${cr},${cg},${cb}, 0.03)`);
        g.addColorStop(1,   `rgba(${cr},${cg},${cb}, 0.00)`);

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
