const storage = require('../../services/storage')
const { recommendAeration } = require('../../services/decision-engine')
const { formatDateTime, round } = require('../../utils/format')

const STATUS_MAP = {
  LOW: { text: '运行平稳', tone: 'safe' },
  MEDIUM: { text: '建议复测', tone: 'warning' },
  HIGH: { text: '存在风险', tone: 'danger' }
}

Page({
  data: {
    latest: null,
    status: STATUS_MAP.MEDIUM,
    decision: null,
    tnRemoval: '--',
    trendSeries: [],
    recordCount: 0,
    alertTitle: '等待数据',
    alertText: '录入一组现场数据后即可生成风险提示。'
  },

  onShow() {
    this.loadDashboard()
  },

  onPullDownRefresh() {
    this.loadDashboard()
    wx.stopPullDownRefresh()
  },

  loadDashboard() {
    const records = storage.getRecords()
    const latest = records[0] || null
    if (!latest) {
      this.setData({ latest: null, recordCount: 0, trendSeries: [] })
      return
    }

    let decision = null
    try {
      decision = recommendAeration(latest)
    } catch (error) {
      console.warn('Dashboard recommendation failed:', error)
    }

    const removal = latest.influentTN > 0
      ? ((latest.influentTN - latest.effluentTN) / latest.influentTN) * 100
      : 0
    const risk = decision ? decision.risk : 'MEDIUM'
    const alert = this.buildAlert(decision)
    const trendSeries = records
      .slice(0, 7)
      .reverse()
      .map((item) => ({
        label: String(item.sampledAt).slice(5, 10),
        value: Number(item.effluentTN)
      }))

    this.setData({
      latest: {
        ...latest,
        displayTime: formatDateTime(latest.sampledAt)
      },
      status: STATUS_MAP[risk],
      decision,
      tnRemoval: round(removal, 1),
      trendSeries,
      recordCount: records.length,
      alertTitle: alert.title,
      alertText: alert.text
    })
  },

  buildAlert(decision) {
    if (!decision) {
      return { title: '数据待核对', text: '最新工况未通过计算，请检查字段与单位。' }
    }
    if (decision.risk === 'HIGH') {
      return {
        title: '先核查，再调整',
        text: decision.actions[0]
      }
    }
    if (decision.risk === 'MEDIUM') {
      return {
        title: '小步试运行',
        text: decision.actions[0]
      }
    }
    return {
      title: '运行状态稳定',
      text: `90% 保守 TN 上界为 ${decision.conservativeUpperTN} mg/L，仍建议按周期复测。`
    }
  },

  goRecord() {
    wx.switchTab({ url: '/pages/record/index' })
  },

  goDecision() {
    wx.switchTab({ url: '/pages/decision/index' })
  },

  goHistory() {
    wx.switchTab({ url: '/pages/history/index' })
  },

  goAbout() {
    wx.navigateTo({ url: '/pages/about/index' })
  }
})
