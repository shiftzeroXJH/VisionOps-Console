import { useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Loader2, Minus, Plus, RefreshCw, X } from 'lucide-react'
import { api } from '../api'

interface Props {
  trial: { trial_id: string; display_name?: string }
  onClose: () => void
}

const metricLabels: Record<string, string> = {
  map50_95: 'mAP50-95',
  map50: 'mAP50',
  precision: 'Precision',
  recall: 'Recall',
}

const MIN_ZOOM = 0.02
const MAX_ZOOM = 12

export function ValidationPreviewDialog({ trial, onClose }: Props) {
  const [imageLimit, setImageLimit] = useState(50)
  const [conf, setConf] = useState(0.25)
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState('')
  const [error, setError] = useState('')
  const [result, setResult] = useState<any>(null)
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [dragStart, setDragStart] = useState<{ x: number; y: number; ox: number; oy: number } | null>(null)
  const [imageIndex, setImageIndex] = useState(0)
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const viewerRef = useRef<HTMLDivElement | null>(null)
  const labelViewportRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!jobId || result || error) return
    const timer = window.setInterval(async () => {
      try {
        const job = await api.getJob(jobId)
        setJobStatus(job.status)
        if (job.status === 'completed') {
          setResult(job.result)
          setImageIndex(0)
          window.clearInterval(timer)
        }
        if (job.status === 'failed') {
          setError(job.error || '验证失败')
          window.clearInterval(timer)
        }
      } catch (err: any) {
        setError(err?.detail?.error || '读取验证任务失败')
        window.clearInterval(timer)
      }
    }, 1200)
    return () => window.clearInterval(timer)
  }, [jobId, result, error])

  const startValidation = async () => {
    setError('')
    setResult(null)
    setJobStatus('queued')
    try {
      const res = await api.validateTrialPreview(trial.trial_id, { image_limit: imageLimit, conf })
      setJobId(res.job_id)
      setJobStatus(res.status)
    } catch (err: any) {
      setError(err?.detail?.error || '启动验证失败')
      setJobStatus('')
    }
  }

  const viewportPoint = (event: React.MouseEvent | React.WheelEvent) => {
    const rect = event.currentTarget.getBoundingClientRect()
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    }
  }

  const viewportCenter = () => {
    const viewport = labelViewportRef.current
    if (!viewport) return null
    return {
      x: viewport.clientWidth / 2,
      y: viewport.clientHeight / 2,
    }
  }

  const fitZoom = () => {
    const viewport = labelViewportRef.current
    if (!viewport || !imageSize.width || !imageSize.height) return 1
    return Math.max(
      MIN_ZOOM,
      Math.min(viewport.clientWidth / imageSize.width, viewport.clientHeight / imageSize.height, MAX_ZOOM)
    )
  }

  const centeredOffset = (nextZoom: number) => {
    const viewport = labelViewportRef.current
    if (!viewport || !imageSize.width || !imageSize.height) return { x: 0, y: 0 }
    return {
      x: (viewport.clientWidth - imageSize.width * nextZoom) / 2,
      y: (viewport.clientHeight - imageSize.height * nextZoom) / 2,
    }
  }

  const adjustZoom = (nextZoom: number, anchor = viewportCenter()) => {
    const boundedZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, Number(nextZoom.toFixed(4))))
    setOffset((current) => {
      if (!anchor) return clampOffset(current, boundedZoom)
      const imageX = (anchor.x - current.x) / zoom
      const imageY = (anchor.y - current.y) / zoom
      return clampOffset({
        x: anchor.x - imageX * boundedZoom,
        y: anchor.y - imageY * boundedZoom,
      }, boundedZoom)
    })
    setZoom(boundedZoom)
  }

  const resetView = () => {
    const nextZoom = fitZoom()
    setZoom(nextZoom)
    setOffset(clampOffset(centeredOffset(nextZoom), nextZoom))
  }

  const imageUrl = (filename: string) => (
    `/api/trials/${trial.trial_id}/validation-previews/${result.validation_id}/files/${filename}`
  )

  const isRunning = Boolean(jobId) && !result && !error
  const images = result?.images || []
  const currentImage = images[imageIndex]

  const moveImage = (delta: number) => {
    if (!images.length) return
    setImageIndex((current) => Math.max(0, Math.min(images.length - 1, current + delta)))
    setImageSize({ width: 0, height: 0 })
  }

  const clampOffset = (candidate: { x: number; y: number }, nextZoom = zoom) => {
    const viewport = labelViewportRef.current
    if (!viewport || !imageSize.width || !imageSize.height) return candidate
    const scaledWidth = imageSize.width * nextZoom
    const scaledHeight = imageSize.height * nextZoom
    const centerX = Math.max(0, (viewport.clientWidth - scaledWidth) / 2)
    const centerY = Math.max(0, (viewport.clientHeight - scaledHeight) / 2)
    const minX = Math.min(0, viewport.clientWidth - scaledWidth)
    const minY = Math.min(0, viewport.clientHeight - scaledHeight)
    return {
      x: scaledWidth <= viewport.clientWidth ? centerX : Math.max(minX, Math.min(0, candidate.x)),
      y: scaledHeight <= viewport.clientHeight ? centerY : Math.max(minY, Math.min(0, candidate.y)),
    }
  }

  useEffect(() => {
    setOffset((current) => clampOffset(current))
  }, [imageSize, zoom, imageIndex])

  useEffect(() => {
    if (!imageSize.width || !imageSize.height) return
    resetView()
  }, [imageSize, imageIndex])

  const renderPane = (title: string, filename: string, isLabel = false) => (
    <section className="validation-pane">
      <div className="validation-pane-title">{title}</div>
      <div
        className="validation-pane-viewport"
        ref={isLabel ? labelViewportRef : undefined}
        onWheel={(event) => {
          event.preventDefault()
          adjustZoom(zoom * (event.deltaY < 0 ? 1.12 : 0.88), viewportPoint(event))
        }}
        onMouseDown={(event) => setDragStart({ x: event.clientX, y: event.clientY, ox: offset.x, oy: offset.y })}
        onMouseMove={(event) => {
          if (!dragStart) return
          setOffset(clampOffset({ x: dragStart.ox + event.clientX - dragStart.x, y: dragStart.oy + event.clientY - dragStart.y }))
        }}
        onMouseUp={() => setDragStart(null)}
        onMouseLeave={() => setDragStart(null)}
      >
        <img
          src={imageUrl(filename)}
          alt={`${title} ${currentImage?.source_image || ''}`}
          draggable={false}
          onLoad={(event) => {
            if (!isLabel) return
            const image = event.currentTarget
            setImageSize({ width: image.naturalWidth, height: image.naturalHeight })
          }}
          style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})` }}
        />
      </div>
    </section>
  )

  return (
    <div className="validation-overlay">
      <div className="validation-dialog">
        <div className="validation-header">
          <div>
            <div className="validation-kicker">Trial 验证</div>
            <h2>{trial.display_name || trial.trial_id}</h2>
          </div>
          <button className="btn" onClick={onClose} title="关闭"><X size={18} /></button>
        </div>

        <div className="validation-toolbar">
          <label>
            图片数量
            <input className="input" type="number" min={1} max={500} value={imageLimit} onChange={(event) => setImageLimit(Number(event.target.value))} disabled={isRunning} />
          </label>
          <label>
            conf
            <input className="input" type="number" min={0.001} max={1} step={0.01} value={conf} onChange={(event) => setConf(Number(event.target.value))} disabled={isRunning} />
          </label>
          <button className="btn btn-primary" onClick={startValidation} disabled={isRunning}>
            {isRunning ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <RefreshCw size={16} />}
            {result ? '重新验证' : '开始验证'}
          </button>
          {isRunning && <span className="text-muted">任务状态：{jobStatus || 'queued'}</span>}
          {error && <span className="text-danger">{error}</span>}
        </div>

        {result && (
          <>
            <div className="validation-metrics">
              {Object.keys(metricLabels).map((key) => (
                <div key={key} className="validation-metric-card">
                  <span>{metricLabels[key]}</span>
                  <strong>{typeof result.metrics?.[key] === 'number' ? result.metrics[key].toFixed(4) : '-'}</strong>
                </div>
              ))}
              <div className="validation-metric-card">
                <span>图片</span>
                <strong>{result.images?.length || 0}</strong>
              </div>
              <div className="validation-metric-note">split: {result.split} · imgsz: {result.imgsz} · conf: {result.conf}</div>
            </div>

            <div className="validation-zoombar">
              <button className="btn" onClick={() => adjustZoom(zoom - 0.25)}><Minus size={15} /></button>
              <span>{Math.round(zoom * 100)}%</span>
              <button className="btn" onClick={() => adjustZoom(zoom + 0.25)}><Plus size={15} /></button>
              <button className="btn" onClick={resetView}>重置视图</button>
              <button className="btn" onClick={() => moveImage(-1)} disabled={imageIndex <= 0}><ChevronLeft size={16} /> 上一张</button>
              <span>{images.length ? `${imageIndex + 1} / ${images.length}` : '0 / 0'}</span>
              <button className="btn" onClick={() => moveImage(1)} disabled={imageIndex >= images.length - 1}>下一张 <ChevronRight size={16} /></button>
              <span className="text-muted">滚轮缩放，拖拽平移；左右两栏同步位置和缩放。</span>
            </div>

            <div className="validation-viewer" ref={viewerRef}>
              {currentImage ? (
                <>
                  <div className="validation-current-caption">{currentImage.source_image || currentImage.label_filename}</div>
                  <div className="validation-pair-grid">
                    {renderPane('Label', currentImage.label_filename || currentImage.filename, true)}
                    {renderPane('Predict', currentImage.predict_filename || currentImage.filename)}
                  </div>
                </>
              ) : (
                <div className="validation-empty">没有可展示的验证图片</div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
