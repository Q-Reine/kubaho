import { useEffect, useState } from "react";

// Reference data-viz palette (light / dark steps of the same hues) and fixed status colours.
const PALETTE = {
  light: {
    surface: "#fcfcfb",
    text: "#0b0b0b",
    textSecondary: "#52514e",
    muted: "#898781",
    grid: "#e1e0d9",
    axis: "#c3c2b7",
    series1: "#2a78d6",
    series2: "#eb6834",
    baseline: "#52514e",
  },
  dark: {
    surface: "#1a1a19",
    text: "#f0efec",
    textSecondary: "#c3c2b7",
    muted: "#898781",
    grid: "#2c2c2a",
    axis: "#383835",
    series1: "#3987e5",
    series2: "#d95926",
    baseline: "#c3c2b7",
  },
};

export function usePalette() {
  const query = "(prefers-color-scheme: dark)";
  const [dark, setDark] = useState(() => window.matchMedia?.(query).matches ?? false);
  useEffect(() => {
    const mq = window.matchMedia?.(query);
    if (!mq) return;
    const onChange = (e) => setDark(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return dark ? PALETTE.dark : PALETTE.light;
}
