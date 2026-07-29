import { useCallback, useEffect, useRef, useState } from 'react'
import { Crop, Maximize2, Minus, Plus, RotateCcw, RotateCw, Trash2 } from 'lucide-react'
import type { Detection, WorkbenchRoi } from '../api'

type Layer = {
  title: string
  boxes: Detection[]
  color?: string
}

interface Props {
  imageUrl: string
  imageName: string
  width: number
  height: number
  layers: Layer[]
  visibleClasses: Set<number>
  showResults: boolean
  roi?: WorkbenchRoi | null
  onRoiChange?: (roi: WorkbenchRoi | null) => void
  onRotateLeft?: () => void
  onRotateRight?: () => void
  controlsDisabled?: boolean
}

const colors = ['#22c55e', '#38bdf8', '#f59e0b', '#f43f5e', '#a78bfa', '#14b8a6', '#e879f9', '#84cc16']

function roiPoints(roi: WorkbenchRoi) {
  const angle = roi.angle * Math.PI / 180
  const ux = { x: Math.cos(angle), y: Math.sin(angle) }
  const uy = { x: -Math.sin(angle), y: Math.cos(angle) }
  const hw = roi.width / 2
  const hh = roi.height / 2
  return [
    [roi.cx - ux.x * hw - uy.x * hh, roi.cy - ux.y * hw - uy.y * hh],
    [roi.cx - ux.x * hw + uy.x * hh, roi.cy - ux.y * hw + uy.y * hh],
    [roi.cx + ux.x * hw + uy.x * hh, roi.cy + ux.y * hw + uy.y * hh],
    [roi.cx + ux.x * hw - uy.x * hh, roi.cy + ux.y * hw - uy.y * hh],
  ] as Array<[number, number]>
}

export function OverlayViewer({
  imageUrl, imageName, width, height, layers, visibleClasses, showResults,
  roi, onRoiChange, onRotateLeft, onRotateRight, controlsDisabled = false,
}: Props) {
  const viewportRef = useRef<HTMLDivElement | null>(null)
  const stageRef = useRef<HTMLDivElement | null>(null)
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [drag, setDrag] = useState<{ pointerId: number; x: number; y: number; ox: number; oy: number } | null>(null)
  const [roiEditing, setRoiEditing] = useState(false)
  const [roiStart, setRoiStart] = useState<{ pointerId: number; x: number; y: number } | null>(null)
  const [draftRoi, setDraftRoi] = useState<WorkbenchRoi | null>(null)

  const fit = useCallback(() => {
    const viewport = viewportRef.current
    if (!viewport || !width || !height) return
    const next = Math.min(viewport.clientWidth / width, viewport.clientHeight / height, 1.5) * 0.94
    setZoom(next)
    setOffset({ x: (viewport.clientWidth - width * next) / 2, y: (viewport.clientHeight - height * next) / 2 })
  }, [height, width])

  useEffect(() => { fit() }, [fit, imageUrl])

  const adjustZoom = (factor: number) => {
    const viewport = viewportRef.current
    if (!viewport) return
    const next = Math.max(0.03, Math.min(12, zoom * factor))
    const cx = viewport.clientWidth / 2
    const cy = viewport.clientHeight / 2
    setOffset({ x: cx - ((cx - offset.x) / zoom) * next, y: cy - ((cy - offset.y) / zoom) * next })
    setZoom(next)
  }

  const imagePoint = (event: React.PointerEvent<HTMLDivElement>) => {
    const stage = stageRef.current
    if (!stage) return null
    const bounds = stage.getBoundingClientRect()
    return {
      x: Math.max(0, Math.min(width, (event.clientX - bounds.left) / bounds.width * width)),
      y: Math.max(0, Math.min(height, (event.clientY - bounds.top) / bounds.height * height)),
    }
  }

  const drawnRoi = (start: { x: number; y: number }, end: { x: number; y: number }): WorkbenchRoi => ({
    cx: (start.x + end.x) / 2,
    cy: (start.y + end.y) / 2,
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
    angle: 0,
  })

  const pointerDown = (event: React.PointerEvent<HTMLDivElement>, layerIndex: number) => {
    if (layerIndex === 0 && roiEditing && onRoiChange) {
      const point = imagePoint(event)
      if (!point) return
      event.currentTarget.setPointerCapture(event.pointerId)
      setRoiStart({ pointerId: event.pointerId, ...point })
      setDraftRoi({ cx: point.x, cy: point.y, width: 0, height: 0, angle: 0 })
      return
    }
    event.currentTarget.setPointerCapture(event.pointerId)
    setDrag({ pointerId: event.pointerId, x: event.clientX, y: event.clientY, ox: offset.x, oy: offset.y })
  }

  const pointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (roiStart?.pointerId === event.pointerId) {
      const point = imagePoint(event)
      if (point) setDraftRoi(drawnRoi(roiStart, point))
      return
    }
    if (drag?.pointerId === event.pointerId) {
      setOffset({ x: drag.ox + event.clientX - drag.x, y: drag.oy + event.clientY - drag.y })
    }
  }

  const pointerEnd = (event: React.PointerEvent<HTMLDivElement>) => {
    if (roiStart?.pointerId === event.pointerId) {
      const point = imagePoint(event)
      const completed = point ? drawnRoi(roiStart, point) : draftRoi
      if (completed && completed.width >= 4 && completed.height >= 4) onRoiChange?.(completed)
      setDraftRoi(null)
      setRoiStart(null)
    }
    if (drag?.pointerId === event.pointerId) setDrag(null)
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  }

  const visibleRoi = draftRoi || roi

  return (
    <div className="overlay-viewer">
      <div className="overlay-viewer-toolbar">
        <span title={imageName}>{imageName}</span>
        <div className="overlay-toolbar-actions">
          {onRoiChange && <div className="roi-tools">
            <button className={`icon-btn ${roiEditing ? 'active' : ''}`} title={roiEditing ? '退出 ROI 绘制' : '绘制 ROI'} disabled={controlsDisabled} onClick={() => setRoiEditing((current) => !current)}><Crop size={16} /></button>
            {roi && <label className="roi-angle" title="ROI 转正角度">
              <span>ROI</span>
              <input type="range" min="-45" max="45" step="1" value={roi.angle} disabled={controlsDisabled} onChange={(event) => onRoiChange({ ...roi, angle: Number(event.target.value) })} />
              <output>{Math.round(roi.angle)}°</output>
            </label>}
            <button className="icon-btn" title="清除 ROI" disabled={controlsDisabled || !roi} onClick={() => onRoiChange(null)}><Trash2 size={16} /></button>
            <button className="icon-btn" title="原图逆时针旋转 90°" disabled={controlsDisabled} onClick={onRotateLeft}><RotateCcw size={16} /></button>
            <button className="icon-btn" title="原图顺时针旋转 90°" disabled={controlsDisabled} onClick={onRotateRight}><RotateCw size={16} /></button>
          </div>}
          <div className="zoom-tools">
            <button className="icon-btn" title="缩小" onClick={() => adjustZoom(0.8)}><Minus size={16} /></button>
            <span>{Math.round(zoom * 100)}%</span>
            <button className="icon-btn" title="放大" onClick={() => adjustZoom(1.25)}><Plus size={16} /></button>
            <button className="icon-btn" title="适应窗口" onClick={fit}><Maximize2 size={16} /></button>
          </div>
        </div>
      </div>
      <div className={`overlay-panes ${layers.length > 1 ? 'paired' : ''}`}>
        {layers.map((layer, layerIndex) => (
          <section className="overlay-pane" key={layer.title}>
            <div className="overlay-pane-title">{layer.title}</div>
            <div
              className={`overlay-viewport ${layerIndex === 0 && roiEditing ? 'roi-editing' : ''}`}
              ref={layerIndex === 0 ? viewportRef : undefined}
              onWheel={(event) => { event.preventDefault(); adjustZoom(event.deltaY < 0 ? 1.12 : 0.88) }}
              onPointerDown={(event) => pointerDown(event, layerIndex)}
              onPointerMove={pointerMove}
              onPointerUp={pointerEnd}
              onPointerCancel={pointerEnd}
            >
              <div ref={layerIndex === 0 ? stageRef : undefined} className="overlay-stage" style={{ width, height, transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})` }}>
                <img src={imageUrl} alt={imageName} draggable={false} />
                {(showResults || (layerIndex === 0 && visibleRoi)) && (
                  <svg viewBox={`0 0 ${width} ${height}`} aria-label={`${layer.title}检测框`}>
                    {showResults && layer.boxes.filter((box) => visibleClasses.has(box.class_id)).map((box, index) => {
                      const color = layer.color || colors[Math.abs(box.class_id) % colors.length]
                      const label = box.confidence == null ? box.class_name : `${box.class_name} ${box.confidence.toFixed(2)}`
                      const labelWidth = Math.max(48, label.length * 8 + 8)
                      const anchor = box.polygon?.[0] || [box.x1, box.y1]
                      const y = Math.max(16, anchor[1])
                      return (
                        <g key={`${box.class_id}-${index}`}>
                          {box.polygon
                            ? <polygon points={box.polygon.map((point) => point.join(',')).join(' ')} fill="none" stroke={color} strokeWidth={3 / zoom} />
                            : <rect x={box.x1} y={box.y1} width={Math.max(0, box.x2 - box.x1)} height={Math.max(0, box.y2 - box.y1)} fill="none" stroke={color} strokeWidth={3 / zoom} />}
                          <rect x={anchor[0]} y={y - 17} width={labelWidth} height={17} fill={color} />
                          <text x={anchor[0] + 4} y={y - 4} fill="#fff" fontSize={12 / Math.max(0.7, zoom)}>{label}</text>
                        </g>
                      )
                    })}
                    {layerIndex === 0 && visibleRoi && visibleRoi.width >= 1 && visibleRoi.height >= 1 && <g className="roi-overlay">
                      <polygon points={roiPoints(visibleRoi).map((point) => point.join(',')).join(' ')} />
                      <circle cx={visibleRoi.cx} cy={visibleRoi.cy} r={5 / zoom} />
                      <text x={visibleRoi.cx + 8 / zoom} y={visibleRoi.cy - 8 / zoom} fontSize={13 / zoom}>ROI {Math.round(visibleRoi.angle)}°</text>
                    </g>}
                  </svg>
                )}
              </div>
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}
