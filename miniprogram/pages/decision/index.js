const config = require('../../config')
const storage = require('../../services/storage')
const modelApi = require('../../services/model-api')
const { formatDateTime } = require('../../utils/format')

const GRADE_META = {
  A: { title: '证据充分', tone: 'safe', summary: '支持域、达标上界和调节步长均通过。' },
  B: { title: '小步试运行', tone: 'warning', summary: '存在安全余量，但需要稳定后复测。' },
  C: { title: '维持并核查', tone: 'danger', summary: '证据或安全余量不足，不输出无约束优化。' }
}

Page({
  data: {
    record: null,
    targetTN: String(config.defaultTargetTN),
    minimumSafeAeration: String(config.defaultMinimumSafeAeration),
    loading: false,
    error: '',
    decision: null,
    gradeMeta: GRADE_META.C,
    sourceLabel: '本地透明引擎'
  },

  onShow() {
    const record = storage.getActiveRecord()
    this.setData({
      record: record ? { ...record, displayTime: formatDateTime(record.sampledAt) } : null,
      decision: null,
      error: ''
    })
    if (record) this.runDecision()
  },

  onOptionInput(event) {
    const field = event.currentTarget.dataset.field
    this.setData({ [field]: event.detail.value })
  },

  runDecision() {
    const record = storage.getActiveRecord()
    if (!record || this.data.loading) return

    const targetTN = Number(this.data.targetTN)
    const minimumSafeAeration = Number(this.data.minimumSafeAeration)
    this.setData({ loading: true, error: '' })

    modelApi.getRecommendation(record, { targetTN, minimumSafeAeration })
      .then(({ source, result }) => {
        const decision = {
          ...result,
          adjustmentText: this.adjustmentText(result),
          supportPercent: Math.round(result.supportScore * 100),
          curve: result.curve || []
        }
        this.setData({
          decision,
          gradeMeta: GRADE_META[result.grade] || GRADE_META.C,
          sourceLabel: source === 'remote-model-api' ? '远程模型 API' : '本地透明引擎',
          loading: false
        })
      })
      .catch((error) => {
        const details = error.details ? Object.values(error.details).join('；') : error.message
        this.setData({ error: details || '决策计算失败', loading: false })
      })
  },

  adjustmentText(result) {
    const delta = Number(result.recommendedAeration) - Number(result.currentAeration)
    if (Math.abs(delta) < 0.05) return '维持当前'
    return delta > 0 ? `上调 ${delta.toFixed(1)}` : `下调 ${Math.abs(delta).toFixed(1)}`
  },

  chooseRecord() {
    wx.switchTab({ url: '/pages/history/index' })
  },

  goRecord() {
    wx.switchTab({ url: '/pages/record/index' })
  },

  copyReport() {
    const decision = this.data.decision
    const record = this.data.record
    if (!decision || !record) return
    const lines = [
      '# WaterPilot 决策摘要',
      `采样时间：${record.displayTime}`,
      `证据等级：${decision.grade}（${this.data.gradeMeta.title}）`,
      `当前/建议曝气：${decision.currentAeration} → ${decision.recommendedAeration} L/min`,
      `预测出水 TN：${decision.predictedEffluentTN} mg/L`,
      `90% 保守上界：${decision.conservativeUpperTN} mg/L`,
      `估算 DO：${decision.estimatedDO} mg/L`,
      `支持域覆盖：${decision.supportPercent}%`,
      '',
      ...decision.actions.map((item, index) => `${index + 1}. ${item}`),
      '',
      '免责声明：科研与教学用途，不替代现场安全联锁与专业人员判断。'
    ]
    wx.setClipboardData({ data: lines.join('\n') })
  }
})
