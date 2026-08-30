const storage = require('../../services/storage')
const { validateCondition } = require('../../services/decision-engine')
const { toDateInput, toTimeInput } = require('../../utils/format')

function emptyForm() {
  const now = new Date()
  return {
    date: toDateInput(now),
    time: toTimeInput(now),
    influentTN: '',
    effluentTN: '',
    cod: '',
    temperature: '25',
    mlss: '',
    currentAeration: '',
    dissolvedOxygen: '',
    ourHetMax: '',
    ourAobMax: '',
    ourNobMax: '',
    ourRealtime: '',
    note: ''
  }
}

Page({
  data: {
    form: emptyForm(),
    errors: [],
    saving: false
  },

  onShow() {
    const active = storage.getActiveRecord()
    if (active && this.data.form.influentTN === '') {
      this.fillFromRecord(active)
    }
  },

  fillFromRecord(record) {
    this.setData({
      form: {
        ...emptyForm(),
        influentTN: String(record.influentTN ?? ''),
        effluentTN: String(record.effluentTN ?? ''),
        cod: String(record.cod ?? ''),
        temperature: String(record.temperature ?? '25'),
        mlss: String(record.mlss ?? ''),
        currentAeration: String(record.currentAeration ?? ''),
        dissolvedOxygen: String(record.dissolvedOxygen ?? ''),
        ourHetMax: String(record.ourHetMax ?? ''),
        ourAobMax: String(record.ourAobMax ?? ''),
        ourNobMax: String(record.ourNobMax ?? ''),
        ourRealtime: String(record.ourRealtime ?? ''),
        note: ''
      }
    })
  },

  onFieldInput(event) {
    const field = event.currentTarget.dataset.field
    this.setData({ [`form.${field}`]: event.detail.value })
  },

  onDateChange(event) {
    this.setData({ 'form.date': event.detail.value })
  },

  onTimeChange(event) {
    this.setData({ 'form.time': event.detail.value })
  },

  saveOnly() {
    this.persist(false)
  },

  saveAndDecide() {
    this.persist(true)
  },

  persist(goDecision) {
    if (this.data.saving) return
    const numericFields = [
      'influentTN', 'effluentTN', 'cod', 'temperature', 'mlss',
      'currentAeration', 'dissolvedOxygen', 'ourHetMax',
      'ourAobMax', 'ourNobMax', 'ourRealtime'
    ]
    const condition = numericFields.reduce((result, field) => {
      result[field] = this.data.form[field]
      return result
    }, {})
    const validation = validateCondition(condition)

    if (!validation.valid) {
      this.setData({ errors: Object.values(validation.errors) })
      wx.showToast({ title: '请核对表单', icon: 'none' })
      return
    }

    this.setData({ saving: true, errors: [] })
    const record = numericFields.reduce((result, field) => {
      result[field] = Number(this.data.form[field])
      return result
    }, {
      id: `record-${Date.now()}`,
      source: 'user',
      sampledAt: `${this.data.form.date}T${this.data.form.time}:00`,
      note: String(this.data.form.note || '').trim()
    })

    storage.saveRecord(record)
    wx.showToast({ title: '已保存到本机', icon: 'success' })
    this.setData({ saving: false, form: emptyForm() })
    if (goDecision) {
      setTimeout(() => wx.switchTab({ url: '/pages/decision/index' }), 350)
    }
  },

  resetForm() {
    wx.showModal({
      title: '清空当前表单？',
      content: '尚未保存的内容将丢失。',
      success: ({ confirm }) => {
        if (confirm) this.setData({ form: emptyForm(), errors: [] })
      }
    })
  }
})
