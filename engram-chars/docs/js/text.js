// text.js — small shared text helpers.

// Remove em/en dashes from a string, turning them into natural comma pauses.
// Used to keep NPC dialogue free of em dashes (a stylistic request). Applied
// only to NPC-spoken text, never to user input or analytical panels.
export function deDash(s) {
  return String(s ?? '')
    .replace(/\s*[—–]\s*/g, ', ')   // em/en dash (with any surrounding spaces) → comma pause
    .replace(/,\s*,/g, ',')          // collapse any doubled commas we just created
    .replace(/\s+([,.!?;:])/g, '$1') // tidy stray space before punctuation
    .replace(/,\s*([.!?;:])/g, '$1'); // drop a comma that now sits before end punctuation
}
