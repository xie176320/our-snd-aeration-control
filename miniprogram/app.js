const storage = require('./services/storage')

App({
  onLaunch() {
    storage.seedDemoRecords()
  },

  globalData: {
    appName: 'WaterPilot 智水云控',
    version: '0.1.0'
  }
})
