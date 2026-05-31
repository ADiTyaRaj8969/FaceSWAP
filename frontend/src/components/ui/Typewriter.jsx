import { useState, useEffect } from 'react';

export default function Typewriter({ words, typingSpeed = 100, deletingSpeed = 50, pauseDuration = 2000 }) {
  const [currentText, setCurrentText] = useState('');
  const [wordIndex, setWordIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    if (!words || words.length === 0) return;
    
    const currentWord = words[wordIndex];
    let timeout;

    if (isDeleting) {
      if (currentText === '') {
        setIsDeleting(false);
        setWordIndex((prev) => (prev + 1) % words.length);
      } else {
        timeout = setTimeout(() => {
          setCurrentText(currentWord.slice(0, currentText.length - 1));
        }, deletingSpeed);
      }
    } else {
      if (currentText === currentWord) {
        timeout = setTimeout(() => {
          setIsDeleting(true);
        }, pauseDuration);
      } else {
        timeout = setTimeout(() => {
          setCurrentText(currentWord.slice(0, currentText.length + 1));
        }, typingSpeed);
      }
    }

    return () => clearTimeout(timeout);
  // We intentionally omit `words` from the dependency array to prevent infinite loops if passed inline
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentText, isDeleting, wordIndex, typingSpeed, deletingSpeed, pauseDuration]);

  return (
    <span className="inline-flex items-center">
      <span className="min-w-[1px]">{currentText}</span>
    </span>
  );
}
