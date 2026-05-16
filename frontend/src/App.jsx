import React, { useCallback, useState } from 'react'
import StlViewer from './components/StlViewer.jsx'
import FeaturePanel from './components/FeaturePanel.jsx'
import DensityChart from './components/DensityChart.jsx'

const EXAMPLE_PROMPTS = [
  'How would this chair look in a hospital?',
  'Convert this into an elderly support chair.',
  'Generate a futuristic gaming version with headrest.',
  'Make this a lounge chair for a cafe.',
  'Adapt this chair for outdoor patio use.',
  'Reconfigure for a wheelchair-accessible setting.',
]

export default function App() {
  const [analysis, setAnalysis] = useState(null)
  const [transformSpec, setTransformSpec] = useState(null)
  const [synth, setSynth] = useState(null)
  const [prompt, setPrompt] = useState(EXAMPLE_PROMPTS[0])
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [wireframe, setWireframe] = useState(false)

  const onUpload = useCallback(async (file) => {
    setError(null); setBusy('analyzing'); setSynth(null); setTransformSpec(null)
    try {
      const fd = new FormData(); fd.append('file', file)
      const res = await fetch('/api/analyze', { method: 'POST', body: fd })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setAnalysis(data)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(null)
    }
  }, [])

  const onTransform = async () => {
    if (!analysis) return
    setError(null); setBusy('transform')
    try {
      const res = await fetch('/api/transform', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: analysis.session_id, prompt }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setTransformSpec(data.spec)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(null)
    }
  }

  const onGenerate = async () => {
    if (!analysis) return
    setError(null); setBusy('generate')
    try {
      const res = await fetch('/api/generate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: analysis.session_id, spec: transformSpec || undefined }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setSynth(data)
      if (!transformSpec) setTransformSpec(data.spec)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(null)
    }
  }

  const onTransformAndGenerate = async () => {
    await onTransform()
    // synth will run with the new spec stored server-side; wait one tick.
    setTimeout(onGenerate, 50)
  }

  const onFileInput = (e) => {
    const f = e.target.files?.[0]
    if (f) onUpload(f)
  }
  const onDrop = (e) => {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) onUpload(f)
  }

  return (
    <div className="app">
      <div className="header">
        <h1>QR-CAD · Synthetic Chair Generation</h1>
        <span className="tag">slicing → QR → NL → synthesis</span>
        <span style={{ flex: 1 }} />
        <label style={{ fontSize: 12, color: '#93a4cf' }}>
          <input type="checkbox" checked={wireframe} onChange={(e) => setWireframe(e.target.checked)} />
          {' '}wireframe
        </label>
      </div>

      <aside className="sidebar">
        <div className="section">
          <h2>1 · Upload STL</h2>
          <label
            className={`dropzone ${dragging ? 'dragging' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <input type="file" accept=".stl" onChange={onFileInput} style={{ display: 'none' }} />
            {busy === 'analyzing'
              ? 'Analyzing…'
              : 'Drop a chair STL here, or click to browse.'}
          </label>
        </div>

        <div className="section">
          <h2>2 · Natural language</h2>
          <div className="field">
            <label>prompt</label>
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          </div>
          <div className="examples">
            {EXAMPLE_PROMPTS.map((p) => (
              <a key={p} onClick={() => setPrompt(p)}>↳ {p}</a>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
            <button className="btn alt" disabled={!analysis || busy} onClick={onTransform}>
              {busy === 'transform' ? '…' : 'Plan'}
            </button>
            <button className="btn" disabled={!analysis || busy} onClick={onTransformAndGenerate}>
              {busy === 'generate' ? 'Generating…' : 'Plan + Generate STL'}
            </button>
          </div>
          {error && <div className="err">{error}</div>}
        </div>

        <div className="section">
          <h2>3 · QR memory</h2>
          {analysis?.qr ? (
            <div>
              <div className="qrwrap">
                <img src={`data:image/png;base64,${analysis.qr.images_b64[0]}`} alt="QR" />
              </div>
              <div className="kv" style={{ marginTop: 8 }}>
                <div className="row"><span className="k">chunks</span><span className="v">{analysis.qr.n_chunks}</span></div>
                <div className="row"><span className="k">raw</span><span className="v">{analysis.qr.raw_bytes} B</span></div>
                <div className="row"><span className="k">compressed</span><span className="v">{analysis.qr.compressed_bytes} B</span></div>
              </div>
              <a className="btn alt" style={{ display: 'inline-block', marginTop: 8, textDecoration: 'none' }}
                 href={analysis.qr.paths?.[0]} download>Download QR</a>
            </div>
          ) : <div className="loading">QR appears after upload.</div>}
        </div>

        {synth && (
          <div className="section">
            <h2>4 · Synthetic STL</h2>
            <a className="btn" style={{ display: 'inline-block', textDecoration: 'none' }}
               href={synth.synthetic_stl_url} download>Download synthetic STL</a>
            <div className="kv" style={{ marginTop: 8 }}>
              <div className="row"><span className="k">triangles</span><span className="v">{synth.triangle_count}</span></div>
              <div className="row"><span className="k">target</span><span className="v">{synth.spec?.target_environment}</span></div>
              <div className="row"><span className="k">source</span><span className="v">{synth.spec?._source || 'rule_based'}</span></div>
            </div>
          </div>
        )}
      </aside>

      <main className="main">
        <div className="viewers">
          <div className="viewer">
            <div className="label">Original {analysis?.source ? `· ${analysis.source}` : ''}</div>
            <StlViewer url={analysis?.original_stl_url} wireframe={wireframe} color="#6f8aff" />
          </div>
          <div className="viewer">
            <div className="label">Synthetic {synth?.spec?.target_environment ? `· ${synth.spec.target_environment}` : ''}</div>
            <StlViewer url={synth?.synthetic_stl_url} wireframe={wireframe} color="#5fd4a4" />
          </div>
        </div>
        <div className="bottom">
          <div className="panel">
            <h3>Extracted features</h3>
            <FeaturePanel intelligence={analysis?.intelligence} />
          </div>
          <div className="panel">
            <h3>Slicing density (Z)</h3>
            <DensityChart values={analysis?.density || []} />
            <div className="kv" style={{ marginTop: 6 }}>
              <div className="row"><span className="k">layers</span><span className="v">{analysis?.density?.length || 0}</span></div>
            </div>
          </div>
          <div className="panel">
            <h3>Transformation spec</h3>
            {transformSpec ? (
              <pre>{JSON.stringify(transformSpec, null, 2)}</pre>
            ) : <div className="loading">Run a prompt to see the plan.</div>}
          </div>
        </div>
      </main>
    </div>
  )
}
