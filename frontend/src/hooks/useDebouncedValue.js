import { useEffect, useState } from "react";

/**
 * Returns `value` delayed by `delay` ms. Each change restarts the timer, so a
 * fast typist produces exactly one settled value instead of one per keystroke.
 *
 * This only controls *when* a value settles — it does not by itself prevent
 * out-of-order responses. Race safety comes from keying the query on the
 * debounced value (see useLiveAnalysis).
 */
export default function useDebouncedValue(value, delay = 500) {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    if (value === debounced) return undefined;
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay, debounced]);

  return debounced;
}
