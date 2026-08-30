Component({
  properties: {
    series: {
      type: Array,
      value: [],
      observer: 'normalizeSeries'
    },
    unit: {
      type: String,
      value: ''
    },
    target: {
      type: Number,
      value: 0
    }
  },

  data: {
    bars: [],
    maximum: 1
  },

  lifetimes: {
    attached() {
      this.normalizeSeries(this.data.series)
    }
  },

  methods: {
    normalizeSeries(series) {
      const clean = (Array.isArray(series) ? series : [])
        .map((item) => ({
          label: String(item.label ?? ''),
          value: Number(item.value)
        }))
        .filter((item) => Number.isFinite(item.value))
      const maximum = Math.max(1, this.data.target || 0, ...clean.map((item) => item.value))
      const bars = clean.map((item) => ({
        ...item,
        displayValue: Number(item.value.toFixed(1)),
        height: Math.max(8, Math.round((item.value / maximum) * 100)),
        overTarget: Boolean(this.data.target && item.value > this.data.target)
      }))
      this.setData({ bars, maximum })
    }
  }
})
