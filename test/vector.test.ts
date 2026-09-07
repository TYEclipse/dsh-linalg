/**
 * Unit tests for the vector core, anchored by the same independent oracle.
 */

import { describe, expect, it } from 'vitest'
import { angleDeg, cross, dot, norm, projection } from '../src/vector.ts'

describe('dot', () => {
  it('3D oracle anchor: (1,2,3)·(4,5,6) = 32', () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBeCloseTo(32, 8)
  })

  it('2D with negatives: (1,−2)·(3,4) = −5', () => {
    expect(dot([1, -2], [3, 4])).toBeCloseTo(-5, 8)
  })

  it('null on length mismatch', () => {
    expect(dot([1, 2], [1, 2, 3])).toBeNull()
  })
})

describe('cross', () => {
  it('3D oracle anchor: (1,2,3)×(4,5,6) = (−3,6,−3)', () => {
    expect(cross([1, 2, 3], [4, 5, 6])).toEqual([-3, 6, -3])
  })

  it('unit vectors: x×y = z', () => {
    expect(cross([1, 0, 0], [0, 1, 0])).toEqual([0, 0, 1])
  })

  it('null when not 3D', () => {
    expect(cross([1, 2], [3, 4])).toBeNull()
    expect(cross([1, 2, 3, 4], [1, 2, 3, 4])).toBeNull()
  })
})

describe('norm', () => {
  it('3D: |(1,2,3)| = √14 ≈ 3.7416573868', () => {
    expect(norm([1, 2, 3])).toBeCloseTo(3.7416573868, 8)
  })

  it('3-4-5 triangle', () => {
    expect(norm([3, 4])).toBeCloseTo(5, 8)
  })
})

describe('projection', () => {
  it('2D onto x-axis: proj((3,4), (1,0)) = (3,0)', () => {
    expect(projection([3, 4], [1, 0])).toEqual([3, 0])
  })

  it('3D onto unit axis: proj((1,1,1), (1,0,0)) = (1,0,0)', () => {
    expect(projection([1, 1, 1], [1, 0, 0])).toEqual([1, 0, 0])
  })

  it('null on zero target vector', () => {
    expect(projection([1, 2], [0, 0])).toBeNull()
  })

  it('null on length mismatch', () => {
    expect(projection([1, 2, 3], [1, 0])).toBeNull()
  })
})

describe('angleDeg', () => {
  it('orthogonal vectors = 90°', () => {
    expect(angleDeg([1, 0], [0, 1])).toBeCloseTo(90, 8)
  })

  it('45° diagonal', () => {
    expect(angleDeg([1, 1], [1, 0])).toBeCloseTo(45, 8)
  })

  it('opposite vectors = 180°', () => {
    expect(angleDeg([1, 0], [-1, 0])).toBeCloseTo(180, 8)
  })

  it('null on zero vector', () => {
    expect(angleDeg([0, 0], [1, 0])).toBeNull()
  })
})
