import { useEffect, useRef } from 'react';
import { motion, useInView, useAnimation } from 'framer-motion';

/* Splits text into individual chars, animates each on mount/scroll */
export default function SplitText({
  text = '',
  className = '',
  delay = 0,
  stagger = 0.04,
  from = { opacity: 0, y: 30 },
  to   = { opacity: 1, y: 0 },
}) {
  const ref     = useRef(null);
  const inView  = useInView(ref, { once: true, margin: '-50px' });
  const controls = useAnimation();

  useEffect(() => {
    if (inView) controls.start('visible');
  }, [inView, controls]);

  const chars = text.split('');

  return (
    <span ref={ref} className={`inline-block ${className}`} aria-label={text}>
      {chars.map((ch, i) => (
        <motion.span
          key={i}
          className="inline-block whitespace-pre"
          variants={{
            hidden:  from,
            visible: { ...to, transition: { duration: 0.5, delay: delay + i * stagger, ease: [0.25, 0.46, 0.45, 0.94] } },
          }}
          initial="hidden"
          animate={controls}
          aria-hidden="true"
        >
          {ch}
        </motion.span>
      ))}
    </span>
  );
}
