const { clamp, round } = require('../utils/format')
const config = require('../config')

const FIELD_RULES = {
  influentTN: { label: '进水 TN', min: 5, max: 150 },
  effluentTN: { label: '出水 TN', min: 0, max: 100 },
  cod: { label: '进水 COD', min: 0, max: 1000 },
  temperature: { label: '温度', min: 5, max: 40 },
  mlss: { label: 'MLSS', min: 0.5, max: 15 },
  currentAeration: { label: '当前曝气量', min: 0.5, max: 30 },
  dissolvedOxygen: { label: 'DO', min: 0, max: 10 },
  ourHetMax: { label: '异养菌最大 OUR', min: 0, max: 100 },
  ourAobMax: { label: 'AOB 最大 OUR', min: 0, max: 50 },
  ourNobMax: { label: 'NOB 最大 OUR', min: 0, max: 50 },
  ourRealtime: { label: '实时 OUR', min: 0, max: 100 }
}

const SUPPORT_RANGES = {
  influentTN: [10, 80],
  cod: [40, 500],
  temperature: [15, 32],
  mlss: [2, 9],
  currentAeration: [3.8, 15],
  dissolvedOxygen: [0.5, 3.5]
}

function toFiniteNumber(value) {
  if (value === '' || value === null || value === undefined) return NaN
  return Number(value)
}

function validateCondition(input) {
  const errors = {}

  Object.entries(FIELD_RULES).forEach(([field, rule]) => {
    const value = toFiniteNumber(input[field])
    if (!Number.isFinite(value)) {
      errors[field] = `${rule.label}不能为空`
    } else if (value < rule.min || value > rule.max) {
      errors[field] = `${rule.label}应在 ${rule.min}–${rule.max} 之间`
    }
  })

  const influent = toFiniteNumber(input.influentTN)
  const effluent = toFiniteNumber(input.effluentTN)
  if (Number.isFinite(influent) && Number.isFinite(effluent) && effluent > influent) {
    errors.effluentTN = '出水 TN 不应高于进水 TN，请核对样品或单位'
  }

  return {
    valid: Object.keys(errors).length === 0,
    errors
  }
}

function normalizeCondition(input) {
  return Object.keys(FIELD_RULES).reduce((result, field) => {
    result[field] = toFiniteNumber(input[field])
    return result
  }, {})
}

function calculateSupport(condition) {
  const fields = Object.keys(SUPPORT_RANGES)
  const outside = fields.filter((field) => {
    const [minimum, maximum] = SUPPORT_RANGES[field]
    return condition[field] < minimum || condition[field] > maximum
  })

  return {
    score: round((fields.length - outside.length) / fields.length, 2),
    outside
  }
}

function calculateActivity(condition) {
  const realtimeRatio = condition.ourHetMax > 0
    ? condition.ourRealtime / condition.ourHetMax
    : 0
  const autotrophCapacity = (condition.ourAobMax + condition.ourNobMax) / 10
  const temperatureFactor = clamp(1 - Math.abs(condition.temperature - 25) * 0.025, 0.55, 1)
  const mlssFactor = clamp(condition.mlss / 6, 0.45, 1.2)

  return clamp(
    (0.38 * realtimeRatio + 0.37 * autotrophCapacity + 0.25 * mlssFactor) * temperatureFactor,
    0,
    1.25
  )
}

function predictAtAeration(condition, aeration, supportScore) {
  const observedRemoval = clamp(
    (condition.influentTN - condition.effluentTN) / condition.influentTN,
    0,
    0.98
  )
  const activity = calculateActivity(condition)
  const delta = aeration - condition.currentAeration
  const estimatedDO = clamp(condition.dissolvedOxygen + delta * 0.18, 0, 6)
  const aerationEffect = delta >= 0 ? 0.0105 * delta : 0.014 * delta
  const activityEffect = clamp((activity - 0.55) * 0.04, -0.035, 0.035)
  const lowDoPenalty = Math.max(0, 0.8 - estimatedDO) * 0.075
  const highDoPenalty = Math.max(0, estimatedDO - 2.3) * 0.022
  const sndBonus = estimatedDO >= 0.8 && estimatedDO <= 1.8
    ? (1 - Math.abs(estimatedDO - 1.3) / 0.5) * 0.012
    : 0
  const removal = clamp(
    observedRemoval + aerationEffect + activityEffect + sndBonus - lowDoPenalty - highDoPenalty,
    0.1,
    0.95
  )
  const predictedEffluentTN = condition.influentTN * (1 - removal)
  const uncertaintyMargin = 1.15 + (1 - supportScore) * 3.2 + Math.abs(delta) * 0.18

  return {
    aeration: round(aeration, 1),
    estimatedDO: round(estimatedDO, 2),
    predictedEffluentTN: round(predictedEffluentTN, 2),
    conservativeUpperTN: round(predictedEffluentTN + uncertaintyMargin, 2),
    removalRate: round(removal * 100, 1)
  }
}

function createCandidateGrid(condition, minimumSafeAeration) {
  const maximum = Math.min(20, Math.max(15, condition.currentAeration + 3))
  const start = Math.ceil(minimumSafeAeration * 5) / 5
  const values = []

  for (let value = start; value <= maximum + 0.001; value += 0.2) {
    values.push(round(value, 1))
  }

  return values
}

function sampleCurve(candidates, desiredPoints = 9) {
  if (candidates.length <= desiredPoints) return candidates
  const sampled = []
  const last = candidates.length - 1
  for (let index = 0; index < desiredPoints; index += 1) {
    sampled.push(candidates[Math.round((index * last) / (desiredPoints - 1))])
  }
  return sampled
}

function recommendAeration(rawCondition, options = {}) {
  const validation = validateCondition(rawCondition)
  if (!validation.valid) {
    const error = new Error('工况数据校验失败')
    error.code = 'INVALID_CONDITION'
    error.details = validation.errors
    throw error
  }

  const condition = normalizeCondition(rawCondition)
  const targetTN = Number(options.targetTN ?? config.defaultTargetTN)
  const minimumSafeAeration = Number(
    options.minimumSafeAeration ?? config.defaultMinimumSafeAeration
  )
  const maxSingleAdjustment = Number(
    options.maxSingleAdjustment ?? config.maxSingleAdjustment
  )

  if (!Number.isFinite(targetTN) || targetTN < 5 || targetTN > 30) {
    throw new RangeError('目标出水 TN 应在 5–30 mg/L 之间')
  }
  if (!Number.isFinite(minimumSafeAeration) || minimumSafeAeration < 0.5 || minimumSafeAeration > 30) {
    throw new RangeError('混合安全下限应在 0.5–30 L/min 之间')
  }

  const support = calculateSupport(condition)
  const activityIndex = calculateActivity(condition)
  const predictions = createCandidateGrid(condition, minimumSafeAeration)
    .map((value) => predictAtAeration(condition, value, support.score))
  const eligible = predictions.filter((item) => (
    item.estimatedDO >= 0.8 &&
    item.estimatedDO <= 2.3 &&
    item.conservativeUpperTN <= targetTN
  ))
  const optimum = eligible.length ? eligible[0] : null
  const fieldTarget = optimum
    ? clamp(
      optimum.aeration,
      condition.currentAeration - maxSingleAdjustment,
      condition.currentAeration + maxSingleAdjustment
    )
    : condition.currentAeration
  const recommendedAeration = round(Math.max(minimumSafeAeration, fieldTarget), 1)
  const recommendation = predictAtAeration(condition, recommendedAeration, support.score)
  const stepLimited = Boolean(optimum && Math.abs(optimum.aeration - recommendedAeration) > 0.05)

  let grade = 'A'
  if (!optimum || support.score < 0.6 || condition.currentAeration < minimumSafeAeration) {
    grade = 'C'
  } else if (support.score < 0.8 || stepLimited) {
    grade = 'B'
  }

  let risk = 'LOW'
  if (grade === 'C' || condition.effluentTN > targetTN) risk = 'HIGH'
  else if (grade === 'B' || recommendation.conservativeUpperTN > targetTN - 1) risk = 'MEDIUM'

  const sndPotential = clamp(
    88 - Math.abs(recommendation.estimatedDO - 1.3) * 24 + (activityIndex - 0.55) * 12,
    25,
    96
  )

  const evidence = [
    `支持域覆盖 ${Math.round(support.score * 100)}%${support.outside.length ? `，越界字段：${support.outside.join('、')}` : ''}`,
    `活性指数 ${round(activityIndex, 2)}，由实时/最大 OUR、自养菌 OUR、温度与 MLSS 综合得到`,
    `建议点预测出水 TN ${recommendation.predictedEffluentTN} mg/L，90% 演示保守上界 ${recommendation.conservativeUpperTN} mg/L`,
    `建议点估算 DO ${recommendation.estimatedDO} mg/L，SND 潜力评分 ${round(sndPotential, 0)}/100`
  ]

  const actions = []
  if (!optimum) {
    actions.push('当前候选范围内没有满足保守 TN 上界的工况，维持当前曝气并优先复测水质与 OUR。')
  } else if (stepLimited) {
    actions.push(`目标点为 ${optimum.aeration} L/min，本次仅调整至 ${recommendedAeration} L/min；稳定 1 个 HRT 后复测。`)
  } else {
    actions.push(`可试运行 ${recommendedAeration} L/min；稳定 1 个 HRT 后复测并重新计算。`)
  }
  if (condition.currentAeration < minimumSafeAeration) {
    actions.push('当前曝气低于混合安全下限，先恢复污泥悬浮与混合条件。')
  }
  if (support.outside.length) {
    actions.push('存在历史支持域外输入，本结果只作风险提示，不作无约束优化。')
  }
  actions.push('膜擦洗风量必须独立核算，现场执行需经专业人员确认。')

  return {
    algorithm: 'transparent-demo-v1',
    generatedAt: new Date().toISOString(),
    grade,
    risk,
    targetTN,
    minimumSafeAeration,
    currentAeration: condition.currentAeration,
    recommendedAeration,
    optimumAeration: optimum ? optimum.aeration : null,
    predictedEffluentTN: recommendation.predictedEffluentTN,
    conservativeUpperTN: recommendation.conservativeUpperTN,
    estimatedDO: recommendation.estimatedDO,
    removalRate: recommendation.removalRate,
    activityIndex: round(activityIndex, 2),
    sndPotential: round(sndPotential, 0),
    supportScore: support.score,
    supportOutside: support.outside,
    stepLimited,
    evidence,
    actions,
    curve: sampleCurve(predictions).map((item) => ({
      label: item.aeration.toFixed(1),
      value: item.conservativeUpperTN
    }))
  }
}

module.exports = {
  FIELD_RULES,
  SUPPORT_RANGES,
  calculateActivity,
  calculateSupport,
  normalizeCondition,
  predictAtAeration,
  recommendAeration,
  validateCondition
}
