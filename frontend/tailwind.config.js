/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    screens: {
      xs:  '480px',
      sm:  '640px',
      md:  '768px',
      lg:  '1024px',
      xl:  '1280px',
      '2xl': '1536px',
    },
    extend: {
      colors: {
        // Premium Light Academic Theme - White Background
        bg:      '#ffffff',   // Pure white background
        bg2:     '#f8fafc',   // Soft pearl for cards
        bg3:     '#f1f5f9',   // Slightly darker grey for contrast areas
        teal:    '#0e7490',   // Marwadi Teal (Deep/Primary)
        'teal-light': '#0891b2', // Marwadi Teal (Lighter/Accent)
        'teal-dark':  '#164e63', // Very dark teal for headings
        gold:    '#c5a55a',   // Shield Gold
        'gold-light': '#d4b872', // Brighter gold
        'gold-dark':  '#a1823f', // Darker gold for contrast on light bg
        navy:    '#0f172a',   // Deep slate for primary text
        'navy-light': '#334155', // Slate for secondary text
        slate:   '#64748b',   // Tertiary text / muted
        cream:   '#0f172a',   // Re-mapped to navy for text contrast in existing classes
        border:  '#e2e8f0',   // Light border
        border2: '#cbd5e1',   // Darker border on hover
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-ring': 'pulse-ring 2s ease-in-out infinite',
        blink:        'blink 2s ease-in-out infinite',
        'float':      'float 6s ease-in-out infinite',
        'glow':       'glow 3s ease-in-out infinite',
      },
      keyframes: {
        'pulse-ring': {
          '0%,100%': { transform: 'scale(1)', opacity: '0.6' },
          '50%':     { transform: 'scale(1.04)', opacity: '1' },
        },
        blink: {
          '0%,100%': { opacity: '1' },
          '50%':     { opacity: '0.3' },
        },
        float: {
          '0%,100%': { transform: 'translateY(0px)' },
          '50%':     { transform: 'translateY(-10px)' },
        },
        glow: {
          '0%,100%': { boxShadow: '0 0 20px rgba(14,116,144,0.15)' },
          '50%':     { boxShadow: '0 0 40px rgba(14,116,144,0.3), 0 0 60px rgba(197,165,90,0.1)' },
        },
      },
    },
  },
  plugins: [],
};
