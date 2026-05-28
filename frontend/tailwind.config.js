/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:      '#0f1209',
        bg2:     '#161a0c',
        bg3:     '#1f2411',
        forest:  '#3D4127',
        olive:   '#636B2F',
        sage:    '#BAC095',
        lime:    '#D4DE95',
        border:  '#3D4127',
        border2: '#4d5535',
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-ring': 'pulse-ring 2s ease-in-out infinite',
        blink:        'blink 2s ease-in-out infinite',
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
      },
    },
  },
  plugins: [],
};
