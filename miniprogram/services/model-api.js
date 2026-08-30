const config = require('../config')
const { recommendAeration } = require('./decision-engine')

function localRecommendation(condition, options) {
  return Promise.resolve({
    source: 'local-transparent-engine',
    result: recommendAeration(condition, options)
  })
}

function requestRemoteRecommendation(condition, options) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: `${config.apiBaseUrl.replace(/\/$/, '')}/v1/recommendations`,
      method: 'POST',
      timeout: config.requestTimeoutMs,
      data: {
        condition,
        target_tn: options.targetTN,
        minimum_safe_aeration: options.minimumSafeAeration
      },
      success(response) {
        if (response.statusCode >= 200 && response.statusCode < 300) {
          resolve({ source: 'remote-model-api', result: response.data })
          return
        }
        reject(new Error(`模型服务返回 ${response.statusCode}`))
      },
      fail(error) {
        reject(new Error(error.errMsg || '模型服务请求失败'))
      }
    })
  })
}

function getRecommendation(condition, options = {}) {
  if (config.demoMode || !config.apiBaseUrl) {
    return localRecommendation(condition, options)
  }
  return requestRemoteRecommendation(condition, options)
}

module.exports = {
  getRecommendation,
  localRecommendation,
  requestRemoteRecommendation
}
