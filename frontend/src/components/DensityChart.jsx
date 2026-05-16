import React from 'react'

export default function DensityChart({ values = [] }) {
  if (!values.length) return <div className="loading">No density profile yet.</div>
  const max = Math.max(...values, 1e-6)
  return (
    <div className="density">
      {values.map((v, i) => (
        <div
          key={i}
          className="bar"
          style={{ height: `${(v / max) * 100}%` }}
          title={`layer ${i + 1}: ${v.toFixed(2)}`}
        />
      ))}
    </div>
  )
}
