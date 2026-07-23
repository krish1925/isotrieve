# Isotrieve Website - Vue.js Application

Modern Vue.js website for the Agent Embedding Communication Protocol.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
isotrieve-website/
├── src/
│   ├── components/      # Reusable Vue components
│   │   ├── NavBar.vue
│   │   ├── Footer.vue
│   │   └── CodeSidebar.vue
│   ├── views/           # Page components
│   │   ├── Home.vue
│   │   ├── Protocol.vue
│   │   ├── Performance.vue
│   │   ├── Playground.vue
│   │   └── Docs.vue
│   ├── router/          # Vue Router configuration
│   │   └── index.js
│   ├── App.vue          # Root component
│   ├── main.js          # Application entry point
│   └── style.css        # Global styles
├── index.html           # HTML template
├── package.json         # Dependencies
└── vite.config.js       # Vite configuration
```

## Features

-  Vue 3 with Composition API
-  Vue Router for navigation
-  Consistent formatting across all components
-  Responsive design
-  Code examples with language toggle
-  Clean, simple styling

## Development

The app runs on `http://localhost:3000` by default.

## Build

Production build outputs to `dist/` directory.
