// Fetches `url` and returns its contents as a string; no path resolution
// or include-handling. Ported from python/shaderutl.py - reading is async
// here since the browser has no synchronous file read.
export async function readfile (url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  return await response.text();
}
