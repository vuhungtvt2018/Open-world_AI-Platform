/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0f1a',
        surface: '#121a2f',
        primary: '#00f0ff',
        secondary: '#ff003c',
        success: '#00ffaa',
        warning: '#ffb300',
        text: '#e2e8f0',
        muted: '#94a3b8',
        border: '#1e293b'
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}
