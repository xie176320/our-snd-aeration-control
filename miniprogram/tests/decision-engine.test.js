const test = require('node:test')
const assert = require('node:assert/strict')

const {
  calculateSupport,
  recommendAeration,
  validateCondition
} = require('../services/decision-engine')

const stableCondition = Object.freeze({
  influentTN: 31.2,
  effluentTN: 12.0,
  cod: 208,
  temperature: 25.1,
  mlss: 5.2,
  currentAeration: 8.2,
  dissolvedOxygen: 1.48,
  ourHetMax: 13.1,
  ourAobMax: 4.22,
  ourNobMax: 2.9,
  ourRealtime: 4.2
})

test('validates a complete condition', () => {
  const result = validateCondition(stableCondition)
  assert.equal(result.valid, true)
  assert.deepEqual(result.errors, {})
})

test('reports missing and out-of-range values with field keys', () => {
  const result = validateCondition({
    ...stableCondition,
    influentTN: '',
    temperature: 99
  })
  assert.equal(result.valid, false)
  assert.match(result.errors.influentTN, /不能为空/)
  assert.match(result.errors.temperature, /5–40/)
})

test('rejects effluent TN above influent TN', () => {
  const result = validateCondition({
    ...stableCondition,
    influentTN: 10,
    effluentTN: 12
  })
  assert.equal(result.valid, false)
  assert.match(result.errors.effluentTN, /不应高于/)
})

test('returns a bounded recommendation for stable conditions', () => {
  const result = recommendAeration(stableCondition)
  assert.ok(['A', 'B'].includes(result.grade))
  assert.ok(result.recommendedAeration >= result.minimumSafeAeration)
  assert.ok(Math.abs(result.recommendedAeration - stableCondition.currentAeration) <= 0.8001)
  assert.ok(result.estimatedDO >= 0)
  assert.ok(result.curve.length >= 7)
  assert.equal(result.algorithm, 'transparent-demo-v1')
})

test('is deterministic except for generated timestamp', () => {
  const first = recommendAeration(stableCondition)
  const second = recommendAeration(stableCondition)
  const { generatedAt: firstTime, ...firstComparable } = first
  const { generatedAt: secondTime, ...secondComparable } = second
  assert.ok(firstTime)
  assert.ok(secondTime)
  assert.deepEqual(firstComparable, secondComparable)
})

test('never recommends below a caller-provided mixing floor', () => {
  const result = recommendAeration(stableCondition, {
    minimumSafeAeration: 7.9
  })
  assert.ok(result.recommendedAeration >= 7.9)
})

test('uses grade C when the current aeration is below the mixing floor', () => {
  const result = recommendAeration(
    { ...stableCondition, currentAeration: 3.2, dissolvedOxygen: 0.45 },
    { minimumSafeAeration: 3.8 }
  )
  assert.equal(result.grade, 'C')
  assert.equal(result.risk, 'HIGH')
  assert.match(result.actions.join(' '), /混合安全下限/)
})

test('uses grade C for conditions far outside the support domain', () => {
  const outside = {
    ...stableCondition,
    influentTN: 120,
    cod: 800,
    temperature: 39,
    mlss: 13,
    currentAeration: 24,
    dissolvedOxygen: 6
  }
  const support = calculateSupport(outside)
  const result = recommendAeration(outside)
  assert.ok(support.score < 0.6)
  assert.equal(result.grade, 'C')
  assert.match(result.evidence[0], /越界字段/)
})

test('maintains current aeration when no conservative candidate can meet target', () => {
  const overloaded = {
    ...stableCondition,
    influentTN: 70,
    effluentTN: 48,
    ourAobMax: 0.5,
    ourNobMax: 0.4,
    ourRealtime: 0.5
  }
  const result = recommendAeration(overloaded, { targetTN: 10 })
  assert.equal(result.grade, 'C')
  assert.equal(result.optimumAeration, null)
  assert.equal(result.recommendedAeration, overloaded.currentAeration)
  assert.match(result.actions[0], /维持当前曝气/)
})

test('throws a typed error for invalid conditions', () => {
  assert.throws(
    () => recommendAeration({ ...stableCondition, mlss: '' }),
    (error) => error.code === 'INVALID_CONDITION' && Boolean(error.details.mlss)
  )
})

test('rejects unsafe target options', () => {
  assert.throws(
    () => recommendAeration(stableCondition, { targetTN: 2 }),
    /5–30/
  )
  assert.throws(
    () => recommendAeration(stableCondition, { minimumSafeAeration: 50 }),
    /0.5–30/
  )
})
