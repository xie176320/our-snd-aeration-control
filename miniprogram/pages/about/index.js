Page({
  data: {
    repositoryUrl: 'https://github.com/xie176320/our-snd-aeration-control',
    flowSteps: ['工况录入', '字段校验', '模型或规则计算', '安全门控', '历史回看']
  },

  copyRepository() {
    wx.setClipboardData({ data: this.data.repositoryUrl })
  }
})
