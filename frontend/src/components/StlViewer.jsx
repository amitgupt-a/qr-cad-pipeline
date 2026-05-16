import React, { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export default function StlViewer({ url, wireframe = false, color = '#6f8aff' }) {
  const mountRef = useRef(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#0b0f17')

    const w = mount.clientWidth || 400
    const h = mount.clientHeight || 400
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 5000)
    camera.position.set(120, 120, 140)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(w, h)
    mount.appendChild(renderer.domElement)

    const grid = new THREE.GridHelper(200, 20, 0x1d2742, 0x11182a)
    scene.add(grid)

    const ambient = new THREE.AmbientLight(0xffffff, 0.55)
    scene.add(ambient)
    const dir = new THREE.DirectionalLight(0xffffff, 0.8)
    dir.position.set(80, 120, 100)
    scene.add(dir)
    const dir2 = new THREE.DirectionalLight(0x88aaff, 0.3)
    dir2.position.set(-80, 30, -60)
    scene.add(dir2)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true

    let mesh = null
    let cancelled = false
    if (url) {
      new STLLoader().load(
        url,
        (geom) => {
          if (cancelled) return
          geom.computeVertexNormals()
          geom.center()
          const material = new THREE.MeshStandardMaterial({
            color: new THREE.Color(color),
            metalness: 0.1,
            roughness: 0.7,
            flatShading: false,
            wireframe,
          })
          mesh = new THREE.Mesh(geom, material)
          // Auto-fit camera.
          geom.computeBoundingBox()
          const bb = geom.boundingBox
          const size = new THREE.Vector3()
          bb.getSize(size)
          const maxDim = Math.max(size.x, size.y, size.z)
          const dist = maxDim * 2.2 + 50
          camera.position.set(dist, dist * 0.8, dist)
          camera.lookAt(0, 0, 0)
          // Stand the chair upright if Z is the up-axis (STL convention varies).
          mesh.rotation.x = -Math.PI / 2
          scene.add(mesh)
        },
        undefined,
        (err) => console.warn('STL load error', err)
      )
    }

    let raf
    const tick = () => {
      controls.update()
      renderer.render(scene, camera)
      raf = requestAnimationFrame(tick)
    }
    tick()

    const onResize = () => {
      const nw = mount.clientWidth
      const nh = mount.clientHeight
      camera.aspect = nw / nh
      camera.updateProjectionMatrix()
      renderer.setSize(nw, nh)
    }
    const ro = new ResizeObserver(onResize)
    ro.observe(mount)

    return () => {
      cancelled = true
      cancelAnimationFrame(raf)
      ro.disconnect()
      controls.dispose()
      renderer.dispose()
      if (mesh) {
        mesh.geometry.dispose()
        mesh.material.dispose()
      }
      mount.removeChild(renderer.domElement)
    }
  }, [url, wireframe, color])

  return <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
}
