import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import '@mantine/dropzone/styles.css';
import '@mantine/core/styles.css';
import App from './App.tsx'
import { MantineProvider } from '@mantine/core'
import { theme } from './theme.ts'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <MantineProvider forceColorScheme='dark' theme={theme}>
      <App />
    </MantineProvider>
  </StrictMode>,
)
