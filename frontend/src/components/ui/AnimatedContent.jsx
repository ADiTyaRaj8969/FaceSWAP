import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';

/* Fade/slide in when element scrolls into view */
export default function AnimatedContent({
  children,
  className = '',
  delay     = 0,
  direction = 'up',   // 'up' | 'down' | 'left' | 'right' | 'none'
  distance  = 30,
  duration  = 0.55,
}) {
  const ref    = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });

  const dirMap = {
    up:    { y: distance },
    down:  { y: -distance },
    left:  { x: distance },
    right: { x: -distance },
    none:  {},
  };

  return (
    <motion.div
      ref={ref}
      className={className}
      initial={{ opacity: 0, ...dirMap[direction] }}
      animate={inView ? { opacity: 1, x: 0, y: 0 } : {}}
      transition={{ duration, delay, ease: [0.25, 0.46, 0.45, 0.94] }}
    >
      {children}
    </motion.div>
  );
}
