import React from 'react'

function Row({ k, v }) {
  return (
    <div className="row">
      <span className="k">{k}</span>
      <span className="v">{v}</span>
    </div>
  )
}

export default function FeaturePanel({ intelligence }) {
  if (!intelligence) return <div className="loading">Upload an STL to extract features.</div>
  const g = intelligence.geometry || {}
  const s = intelligence.slicing || {}
  const c = intelligence.classification || {}
  const dim = g.dimensions || {}
  return (
    <div className="kv">
      <Row k="source" v={intelligence.source} />
      <Row k="triangles" v={g.triangle_count} />
      <Row k="bbox (cm)" v={`${dim.x?.toFixed(1)} × ${dim.y?.toFixed(1)} × ${dim.z?.toFixed(1)}`} />
      <Row k="surface area" v={g.surface_area?.toFixed(1)} />
      <Row k="volume" v={g.volume?.toFixed(1)} />
      <Row k="seat height" v={s.seat_height?.toFixed(1)} />
      <Row k="base type" v={s.base_type} />
      <Row k="leg count" v={s.leg_count} />
      <Row k="armrests" v={String(s.armrests)} />
      <Row k="backrest angle" v={s.backrest_angle_deg ?? '—'} />
      <Row k="stability" v={s.stability?.toFixed(2)} />
      <Row k="ergonomic" v={s.ergonomic_score?.toFixed(2)} />
      <Row k="slice layers" v={s.slice_layers} />
      <Row k="classification" v={`${c.label} (${(c.confidence * 100).toFixed(0)}%)`} />
      <div className="chips" style={{ marginTop: 8 }}>
        {c.scores && Object.entries(c.scores).map(([k, v]) =>
          v > 0 ? <span key={k} className={`chip ${k === c.label ? 'active' : ''}`}>
            {k} {(v * 100).toFixed(0)}%
          </span> : null
        )}
      </div>
    </div>
  )
}
