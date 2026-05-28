import { useRef, useState } from 'react';
import { motion } from 'framer-motion';

/* Magnetic button effect -- child follows cursor on hover */
export default function Magnet({ children, strength = 0.35, className = '' }) {
  const ref  = useRef(null);
  const [delta, setDelta] = useState({ x: 0, y: 0 });

  const handleMove = (e) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const cx = rect.left + rect.width  / 2;
    const cy = rect.top  + rect.height / 2;
    setDelta({
      x: (e.clientX - cx) * strength,
      y: (e.clientY - cy) * strength,
    });
  };
  const reset = () => setDelta({ x: 0, y: 0 });

  return (
    <motion.div
      ref={ref}
      className={`inline-block ${className}`}
      onMouseMove={handleMove}
      onMouseLeave={reset}
      animate={{ x: delta.x, y: delta.y }}
      transition={{ type: 'spring', stiffness: 200, damping: 20, mass: 0.5 }}
    >
      {children}
    </motion.div>
  );
}
