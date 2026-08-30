function pad(value) {
  return String(value).padStart(2, '0')
}

function toDateInput(date = new Date()) {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`
}

function toTimeInput(date = new Date()) {
  return `${pad(date.getHours())}:${pad(date.getMinutes())}`
}

function formatDateTime(value) {
  if (!value) return '--'
  const normalized = String(value).replace('T', ' ')
  return normalized.slice(0, 16)
}

function round(value, digits = 1) {
  const factor = 10 ** digits
  return Math.round((Number(value) + Number.EPSILON) * factor) / factor
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value))
}

function escapeCsv(value) {
  const text = value === null || value === undefined ? '' : String(value)
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

module.exports = {
  clamp,
  escapeCsv,
  formatDateTime,
  round,
  toDateInput,
  toTimeInput
}
