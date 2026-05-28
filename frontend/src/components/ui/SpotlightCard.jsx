import { useRef, useState } from 'react';

/* Card with a radial spotlight that follows the mouse -- React Bits style */
export default function SpotlightCard({ children, className = '', spotColor = 'rgba(99,107,47,0.18)' }) {
  const cardRef = useRef(null);
  const [pos, setPos]     = useState({ x: 0, y: 0 });
  const [visible, setVisible] = useState(false);

  const handleMove = (e) => {
    const rect = cardRef.current?.getBoundingClientRect();
    if (!rect) return;
    setPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMove}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      className={`relative overflow-hidden ${className}`}
      style={{
        background: visible
          ? `radial-gradient(400px circle at ${pos.x}px ${pos.y}px, ${spotColor}, transparent 60%)`
          : 'transparent',
        transition: 'background 0.1s ease',
      }}
    >
      {children}
    </div>
  );
}
