const { toDateInput } = require('../utils/format')

const DEMO_SERIES = [
  { influentTN: 34.8, effluentTN: 14.3, cod: 238, doValue: 1.18, aeration: 7.6, mlss: 6.2, realtime: 2.8 },
  { influentTN: 33.1, effluentTN: 13.5, cod: 226, doValue: 1.24, aeration: 7.8, mlss: 6.0, realtime: 3.2 },
  { influentTN: 35.6, effluentTN: 14.8, cod: 251, doValue: 1.05, aeration: 7.4, mlss: 5.8, realtime: 2.6 },
  { influentTN: 31.9, effluentTN: 12.7, cod: 219, doValue: 1.42, aeration: 8.0, mlss: 5.7, realtime: 3.7 },
  { influentTN: 32.5, effluentTN: 12.2, cod: 212, doValue: 1.51, aeration: 8.1, mlss: 5.5, realtime: 4.1 },
  { influentTN: 30.8, effluentTN: 11.8, cod: 205, doValue: 1.56, aeration: 8.2, mlss: 5.3, realtime: 4.4 },
  { influentTN: 31.2, effluentTN: 12.0, cod: 208, doValue: 1.48, aeration: 8.2, mlss: 5.2, realtime: 4.2 }
]

function createDemoRecords(baseDate = new Date()) {
  return DEMO_SERIES.map((item, index) => {
    const day = new Date(baseDate)
    day.setHours(9, 30, 0, 0)
    day.setDate(day.getDate() - (DEMO_SERIES.length - 1 - index))

    return {
      id: `demo-${toDateInput(day)}`,
      source: 'demo',
      sampledAt: `${toDateInput(day)}T09:30:00`,
      influentTN: item.influentTN,
      effluentTN: item.effluentTN,
      cod: item.cod,
      temperature: 24.5 + index * 0.1,
      mlss: item.mlss,
      currentAeration: item.aeration,
      dissolvedOxygen: item.doValue,
      ourHetMax: 14.6 - index * 0.25,
      ourAobMax: 4.7 - index * 0.08,
      ourNobMax: 3.2 - index * 0.05,
      ourRealtime: item.realtime,
      note: index === DEMO_SERIES.length - 1 ? '演示数据：系统运行平稳' : ''
    }
  })
}

module.exports = {
  createDemoRecords
}
