import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';

/* Animates text word-by-word with blur + fade */
export default function BlurText({ text = '', className = '', delay = 0 }) {
  const ref    = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-40px' });
  const words  = text.split(' ');

  return (
    <span ref={ref} className={`inline ${className}`}>
      {words.map((word, i) => (
        <motion.span
          key={i}
          className="inline-block mr-[0.25em]"
          initial={{ opacity: 0, filter: 'blur(12px)', y: 8 }}
          animate={inView ? { opacity: 1, filter: 'blur(0px)', y: 0 } : {}}
          transition={{ duration: 0.6, delay: delay + i * 0.07, ease: 'easeOut' }}
        >
          {word}
        </motion.span>
      ))}
    </span>
  );
}
