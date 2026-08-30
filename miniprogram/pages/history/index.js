const storage = require('../../services/storage')
const { formatDateTime, round } = require('../../utils/format')

const FILTERS = [
  { key: 'all', label: '全部' },
  { key: 'safe', label: '已达标' },
  { key: 'risk', label: '需关注' }
]

Page({
  data: {
    filters: FILTERS,
    activeFilter: 'all',
    allRecords: [],
    records: [],
    targetTN: 15
  },

  onShow() {
    this.loadRecords()
  },

  loadRecords() {
    const allRecords = storage.getRecords().map((record) => {
      const removal = record.influentTN > 0
        ? ((record.influentTN - record.effluentTN) / record.influentTN) * 100
        : 0
      const safe = Number(record.effluentTN) <= this.data.targetTN
      return {
        ...record,
        displayTime: formatDateTime(record.sampledAt),
        removalRate: round(removal, 1),
        safe,
        statusText: safe ? '已达标' : '需关注',
        statusTone: safe ? 'safe' : 'danger'
      }
    })
    this.setData({ allRecords }, () => this.applyFilter())
  },

  setFilter(event) {
    this.setData({ activeFilter: event.currentTarget.dataset.key }, () => this.applyFilter())
  },

  applyFilter() {
    const filter = this.data.activeFilter
    const records = this.data.allRecords.filter((record) => {
      if (filter === 'safe') return record.safe
      if (filter === 'risk') return !record.safe
      return true
    })
    this.setData({ records })
  },

  useRecord(event) {
    const id = event.currentTarget.dataset.id
    storage.setActiveRecord(id)
    wx.switchTab({ url: '/pages/decision/index' })
  },

  deleteRecord(event) {
    const id = event.currentTarget.dataset.id
    wx.showModal({
      title: '删除这条记录？',
      content: '删除后无法恢复。',
      confirmColor: '#B73B48',
      success: ({ confirm }) => {
        if (!confirm) return
        storage.removeRecord(id)
        this.loadRecords()
        wx.showToast({ title: '已删除', icon: 'none' })
      }
    })
  },

  copyCsv() {
    if (!this.data.allRecords.length) return
    wx.setClipboardData({
      data: storage.recordsToCsv(this.data.allRecords),
      success: () => wx.showToast({ title: 'CSV 已复制', icon: 'success' })
    })
  },

  resetDemo() {
    wx.showModal({
      title: '恢复演示数据？',
      content: '当前本机记录会被清空，并重新生成 7 天虚构演示数据。',
      confirmColor: '#087F72',
      success: ({ confirm }) => {
        if (!confirm) return
        storage.clearRecords()
        storage.seedDemoRecords()
        this.loadRecords()
      }
    })
  },

  goRecord() {
    wx.switchTab({ url: '/pages/record/index' })
  }
})
