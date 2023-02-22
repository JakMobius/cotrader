import React from 'react';
import { createRoot } from 'react-dom/client';
import "./style.css";
import App from './components/App';

document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('root');
    const root = createRoot(container!);
    root.render(React.createElement(App));
});
