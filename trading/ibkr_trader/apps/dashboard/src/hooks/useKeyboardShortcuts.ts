'use client';
import { useEffect } from 'react';
import { SYSTEM_ORDER } from '@/lib/types';

interface Options {
  onRefresh: () => void;
}

export function useKeyboardShortcuts({ onRefresh }: Options) {
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      // Skip if user is typing in an input
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      switch (e.key.toLowerCase()) {
        case 'r': {
          e.preventDefault();
          onRefresh();
          break;
        }
        case '1':
        case '2':
        case '3': {
          e.preventDefault();
          const idx = Number(e.key) - 1;
          const sys = SYSTEM_ORDER[idx];
          if (sys) {
            const el = document.getElementById(`system-${sys}`);
            el?.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
          break;
        }
      }
    }

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onRefresh]);
}
