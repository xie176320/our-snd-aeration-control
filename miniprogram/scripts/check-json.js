const fs = require('node:fs')
const path = require('node:path')

const root = path.resolve(__dirname, '..')
const ignored = new Set(['node_modules'])

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    if (ignored.has(entry.name)) return []
    const target = path.join(directory, entry.name)
    if (entry.isDirectory()) return walk(target)
    return entry.isFile() && entry.name.endsWith('.json') ? [target] : []
  })
}

const files = walk(root)
files.forEach((file) => {
  JSON.parse(fs.readFileSync(file, 'utf8'))
})

console.log(`Validated ${files.length} JSON files.`)
