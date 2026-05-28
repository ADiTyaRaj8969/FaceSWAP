import { useRef, useState } from 'react';
import { motion } from 'framer-motion';

/* Card that tilts based on mouse position */
export default function TiltedCard({ children, className = '', maxTilt = 8 }) {
  const ref = useRef(null);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });

  const handleMove = (e) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const px = (e.clientX - rect.left) / rect.width  - 0.5;
    const py = (e.clientY - rect.top)  / rect.height - 0.5;
    setTilt({ x: -py * maxTilt, y: px * maxTilt });
  };
  const reset = () => setTilt({ x: 0, y: 0 });

  return (
    <motion.div
      ref={ref}
      className={`${className}`}
      onMouseMove={handleMove}
      onMouseLeave={reset}
      animate={{ rotateX: tilt.x, rotateY: tilt.y }}
      transition={{ type: 'spring', stiffness: 180, damping: 22 }}
      style={{ transformStyle: 'preserve-3d', perspective: 800 }}
    >
      {children}
    </motion.div>
  );
}
