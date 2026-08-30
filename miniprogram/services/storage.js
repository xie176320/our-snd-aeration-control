const { createDemoRecords } = require('../data/demo')
const { escapeCsv } = require('../utils/format')

const RECORDS_KEY = 'waterpilot.records.v1'
const ACTIVE_RECORD_KEY = 'waterpilot.activeRecordId.v1'

function getRecords() {
  const records = wx.getStorageSync(RECORDS_KEY)
  if (!Array.isArray(records)) return []
  return records.slice().sort((a, b) => String(b.sampledAt).localeCompare(String(a.sampledAt)))
}

function saveRecord(record) {
  const records = getRecords().filter((item) => item.id !== record.id)
  const next = [record, ...records]
  wx.setStorageSync(RECORDS_KEY, next)
  wx.setStorageSync(ACTIVE_RECORD_KEY, record.id)
  return record
}

function removeRecord(id) {
  const next = getRecords().filter((item) => item.id !== id)
  wx.setStorageSync(RECORDS_KEY, next)
  if (wx.getStorageSync(ACTIVE_RECORD_KEY) === id) {
    wx.removeStorageSync(ACTIVE_RECORD_KEY)
  }
}

function clearRecords() {
  wx.removeStorageSync(RECORDS_KEY)
  wx.removeStorageSync(ACTIVE_RECORD_KEY)
}

function seedDemoRecords() {
  const records = getRecords()
  if (records.length) return false
  const demos = createDemoRecords()
  wx.setStorageSync(RECORDS_KEY, demos)
  wx.setStorageSync(ACTIVE_RECORD_KEY, demos[demos.length - 1].id)
  return true
}

function setActiveRecord(id) {
  wx.setStorageSync(ACTIVE_RECORD_KEY, id)
}

function getActiveRecord() {
  const records = getRecords()
  const activeId = wx.getStorageSync(ACTIVE_RECORD_KEY)
  return records.find((item) => item.id === activeId) || records[0] || null
}

function recordsToCsv(records = getRecords()) {
  const columns = [
    ['sampledAt', '采样时间'],
    ['influentTN', '进水TN(mg/L)'],
    ['effluentTN', '出水TN(mg/L)'],
    ['cod', '进水COD(mg/L)'],
    ['temperature', '温度(℃)'],
    ['mlss', 'MLSS(g/L)'],
    ['currentAeration', '曝气量(L/min)'],
    ['dissolvedOxygen', 'DO(mg/L)'],
    ['ourHetMax', '异养菌最大OUR'],
    ['ourAobMax', 'AOB最大OUR'],
    ['ourNobMax', 'NOB最大OUR'],
    ['ourRealtime', '实时OUR'],
    ['note', '备注']
  ]
  const lines = [columns.map(([, label]) => escapeCsv(label)).join(',')]
  records.forEach((record) => {
    lines.push(columns.map(([key]) => escapeCsv(record[key])).join(','))
  })
  return `\uFEFF${lines.join('\n')}`
}

module.exports = {
  clearRecords,
  getActiveRecord,
  getRecords,
  recordsToCsv,
  removeRecord,
  saveRecord,
  seedDemoRecords,
  setActiveRecord
}
