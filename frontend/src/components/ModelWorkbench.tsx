import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ActivitySquare, CheckCircle2, ChevronDown, ChevronUp, FolderOpen, Home, ImagePlus, Loader2, Play, PlayCircle, ScanSearch, Settings, Trash2 } from 'lucide-react'
import { api, type Detection, type WorkbenchImage, type WorkbenchModel, type WorkbenchRoi } from '../api'
import { ModelSelector, type ModelSelection } from './ModelSelector'
import { OverlayViewer } from './OverlayViewer'
import { SettingsDialog } from './SettingsDialog'

interface Props { tab: 'inference' | 'evaluation' }

const defaultModel: ModelSelection = { model_source: 'platform', trial_id: '', checkpoint_name: '', model_path: '' }
type ClassInfo = { class_id: number; class_name: string }
type InferenceSession = { session_id: string; images: WorkbenchImage[]; classes: ClassInfo[] }
type DatasetInspection = { dataset_type: string; image_count: number; classes: Array<{ class_id?: number; class_name: string }> }
type MetricRow = ClassInfo & { map50?: number; map50_95?: number; precision?: number; recall?: number }
type EvaluationResult = {
  evaluation_id: string
  images: WorkbenchImage[]
  classes: ClassInfo[]
  metrics: Record<string, number>
  per_class_metrics: MetricRow[]
  predictions_dir: string
}
type JobResult<T> = { status: string; result?: T; error?: string }
type SidebarSide = 'left' | 'right'

const sidebarLimits = {
  leftMin: 140,
  leftMax: 360,
  rightMin: 160,
  rightMax: 360,
  centerMin: 320,
}

function clampSidebarWidths(left: number, right: number, total: number) {
  const usable = Math.max(0, total - sidebarLimits.centerMin)
  const nextLeft = Math.min(Math.max(left, sidebarLimits.leftMin), sidebarLimits.leftMax, Math.max(sidebarLimits.leftMin, usable - sidebarLimits.rightMin))
  const nextRight = Math.min(Math.max(right, sidebarLimits.rightMin), sidebarLimits.rightMax, Math.max(sidebarLimits.rightMin, usable - nextLeft))
  return { left: nextLeft, right: nextRight }
}

function useResizableSidebars() {
  const gridRef = useRef<HTMLDivElement | null>(null)
  const dragRef = useRef<{ side: SidebarSide; pointerId: number; startX: number; left: number; right: number } | null>(null)
  const [widths, setWidths] = useState({ left: 208, right: 220 })
  const setGridRef = useCallback((node: HTMLDivElement | null) => { gridRef.current = node }, [])

  useEffect(() => {
    const grid = gridRef.current
    if (!grid || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(() => {
      if (grid.clientWidth <= 620) return
      setWidths((current) => {
        const next = clampSidebarWidths(current.left, current.right, grid.clientWidth)
        return next.left === current.left && next.right === current.right ? current : next
      })
    })
    observer.observe(grid)
    return () => observer.disconnect()
  }, [])

  const resize = (left: number, right: number) => {
    const total = gridRef.current?.clientWidth || 0
    if (total <= 620) return
    setWidths(clampSidebarWidths(left, right, total))
  }

  const handlePointerDown = (side: SidebarSide, event: React.PointerEvent<HTMLDivElement>) => {
    if ((gridRef.current?.clientWidth || 0) <= 620) return
    event.currentTarget.setPointerCapture(event.pointerId)
    dragRef.current = { side, pointerId: event.pointerId, startX: event.clientX, ...widths }
  }

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    const delta = event.clientX - drag.startX
    resize(drag.side === 'left' ? drag.left + delta : drag.left, drag.side === 'right' ? drag.right - delta : drag.right)
  }

  const handlePointerEnd = (event: React.PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId !== event.pointerId) return
    dragRef.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  }

  const handleKeyDown = (side: SidebarSide, event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
    event.preventDefault()
    const step = event.shiftKey ? 32 : 12
    const direction = event.key === 'ArrowRight' ? 1 : -1
    resize(side === 'left' ? widths.left + direction * step : widths.left, side === 'right' ? widths.right - direction * step : widths.right)
  }

  return { setGridRef, widths, handlePointerDown, handlePointerMove, handlePointerEnd, handleKeyDown }
}

type SidebarResizeHandlesProps = Pick<ReturnType<typeof useResizableSidebars>, 'widths' | 'handlePointerDown' | 'handlePointerMove' | 'handlePointerEnd' | 'handleKeyDown'>

function SidebarResizeHandles({ widths, handlePointerDown, handlePointerMove, handlePointerEnd, handleKeyDown }: SidebarResizeHandlesProps) {
  return <>
    {(['left', 'right'] as const).map((side) => {
      const value = widths[side]
      return <div
        key={side}
        className={`sidebar-resize-handle ${side}`}
        style={side === 'left' ? { left: value } : { right: value }}
        role="separator"
        aria-label={side === 'left' ? '调整图片栏宽度' : '调整类别栏宽度'}
        aria-orientation="vertical"
        aria-valuemin={side === 'left' ? sidebarLimits.leftMin : sidebarLimits.rightMin}
        aria-valuemax={side === 'left' ? sidebarLimits.leftMax : sidebarLimits.rightMax}
        aria-valuenow={value}
        tabIndex={0}
        onPointerDown={(event) => handlePointerDown(side, event)}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerEnd}
        onPointerCancel={handlePointerEnd}
        onKeyDown={(event) => handleKeyDown(side, event)}
      />
    })}
  </>
}

const errorMessage = (error: unknown, fallback: string) => {
  if (typeof error !== 'object' || error === null) return fallback
  const value = error as { detail?: { error?: string }; message?: string }
  return value.detail?.error || value.message || fallback
}

function ClassFilter({ classes, boxes, visible, onChange }: { classes: ClassInfo[]; boxes: Detection[]; visible: Set<number>; onChange: (next: Set<number>) => void }) {
  const counts = useMemo(() => {
    const next = new Map<number, number>()
    boxes.forEach((box) => next.set(box.class_id, (next.get(box.class_id) || 0) + 1))
    return next
  }, [boxes])
  const allVisible = classes.length > 0 && classes.every((item) => visible.has(item.class_id))
  return (
    <aside className="class-panel">
      <div className="class-panel-title">
        <span>类别</span>
        <div className="class-panel-actions">
          <span>{boxes.length} 个目标</span>
          <label title={allVisible ? '隐藏所有类别' : '显示所有类别'}>
            <input
              type="checkbox"
              checked={allVisible}
              disabled={!classes.length}
              onChange={(event) => onChange(event.target.checked
                ? new Set(classes.map((item) => item.class_id))
                : new Set())}
            />
            <span>全部</span>
          </label>
        </div>
      </div>
      <div className="class-list">
        {classes.map((item) => (
          <label key={item.class_id} className="class-row">
            <input type="checkbox" checked={visible.has(item.class_id)} onChange={(event) => {
              const next = new Set(visible)
              if (event.target.checked) next.add(item.class_id)
              else next.delete(item.class_id)
              onChange(next)
            }} />
            <span className="class-swatch" style={{ background: `hsl(${(item.class_id * 67 + 145) % 360} 68% 48%)` }} />
            <span title={item.class_name}>{item.class_name}</span>
            <strong>{counts.get(item.class_id) || 0}</strong>
          </label>
        ))}
        {!classes.length && <div className="workbench-empty-small">运行模型后显示类别</div>}
      </div>
    </aside>
  )
}

function ImageRail({ images, currentId, imageUrl, onSelect, selectedIds, onSelectionChange, onDeleteSelected, deleteDisabled }: { images: WorkbenchImage[]; currentId: string; imageUrl: (item: WorkbenchImage) => string; onSelect: (id: string) => void; selectedIds?: Set<string>; onSelectionChange?: (next: Set<string>) => void; onDeleteSelected?: () => void; deleteDisabled?: boolean }) {
  const activeSelectedIds = selectedIds || new Set<string>()
  const selectable = Boolean(onSelectionChange)
  const allSelected = selectable && images.length > 0 && images.every((item) => activeSelectedIds.has(item.image_id))
  return (
    <aside className="image-rail">
      <div className="image-rail-title">
        <span className="image-rail-heading">图片 <i>{images.length}</i></span>
        {selectable && <div className="image-rail-actions">
          <label title={allSelected ? '取消全选' : '选择全部图片'}>
            <input type="checkbox" checked={allSelected} disabled={!images.length} onChange={(event) => onSelectionChange?.(event.target.checked ? new Set(images.map((item) => item.image_id)) : new Set())} />
            <span>全选</span>
          </label>
          <button className="image-rail-delete" title={activeSelectedIds.size ? `删除选中的 ${activeSelectedIds.size} 张图片` : '请先选择图片'} disabled={deleteDisabled || !activeSelectedIds.size} onClick={onDeleteSelected}><Trash2 size={14} /></button>
        </div>}
      </div>
      <div className="image-rail-list">
        {images.map((item) => (
          <div key={item.image_id} role="button" tabIndex={0} className={`image-rail-item ${selectable ? 'selectable' : ''} ${currentId === item.image_id ? 'active' : ''}`} onClick={() => onSelect(item.image_id)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') onSelect(item.image_id) }} title={item.name}>
            {selectable && <input type="checkbox" aria-label={`选择 ${item.name}`} checked={activeSelectedIds.has(item.image_id)} onClick={(event) => event.stopPropagation()} onChange={(event) => {
              const next = new Set(activeSelectedIds)
              if (event.target.checked) next.add(item.image_id)
              else next.delete(item.image_id)
              onSelectionChange?.(next)
            }} />}
            <img src={imageUrl(item)} alt="" />
            <span>{item.name}</span>
            <i className={`image-status ${item.status || 'completed'}`} title={item.error || item.status} />
          </div>
        ))}
      </div>
    </aside>
  )
}

export function ModelWorkbench({ tab }: Props) {
  const initialEvaluation = useMemo(() => {
    const query = window.location.hash.split('?', 2)[1] || ''
    const params = new URLSearchParams(query)
    const parsedImgsz = Number(params.get('imgsz') || 0)
    return {
      trialId: params.get('trial_id') || '',
      datasetPath: params.get('dataset_path') || '',
      imgsz: Number.isFinite(parsedImgsz) && parsedImgsz > 0 ? parsedImgsz : 640,
    }
  }, [])
  const [models, setModels] = useState<WorkbenchModel[]>([])
  const [model, setModel] = useState<ModelSelection>(() => ({ ...defaultModel, trial_id: initialEvaluation.trialId }))
  const [pythonPath, setPythonPath] = useState('')
  const [showSettings, setShowSettings] = useState(false)

  useEffect(() => {
    api.getWorkbenchModels().then((result) => {
      setModels(result.models || [])
      setPythonPath(result.effective_yolo_python || '')
      setModel((current) => {
        if (!current.trial_id) return current
        const selected = result.models?.find((item) => item.trial_id === current.trial_id)
        return selected && !current.checkpoint_name
          ? { ...current, checkpoint_name: selected.default_checkpoint }
          : current
      })
    }).catch(console.error)
  }, [])

  return (
    <div className="workbench-page">
      <header className="workbench-header">
        <button className="icon-btn" title="返回首页" onClick={() => { window.location.hash = '#/' }}><Home size={18} /></button>
        <div className="workbench-brand"><ScanSearch size={20} /><span>模型工作台</span></div>
        <nav className="workbench-tabs">
          <button className={tab === 'inference' ? 'active' : ''} onClick={() => { window.location.hash = '#/workbench/inference' }}>图片推理</button>
          <button className={tab === 'evaluation' ? 'active' : ''} onClick={() => { window.location.hash = '#/workbench/evaluation' }}>模型评估</button>
        </nav>
        <span className="python-indicator" title={pythonPath}>YOLO Python · {pythonPath ? '已配置' : '未检测到'}</span>
        <button className="icon-btn" title="全局设置" onClick={() => setShowSettings(true)}><Settings size={18} /></button>
      </header>
      {tab === 'inference'
        ? <InferenceView models={models} model={model} onModelChange={setModel} />
        : <EvaluationView models={models} model={model} onModelChange={setModel} initialDatasetPath={initialEvaluation.datasetPath} initialImgsz={initialEvaluation.imgsz} />}
      {showSettings && <SettingsDialog onClose={() => setShowSettings(false)} />}
    </div>
  )
}

function modelPayload(model: ModelSelection) {
  return model.model_source === 'platform'
    ? { model_source: 'platform', trial_id: model.trial_id, checkpoint_name: model.checkpoint_name }
    : { model_source: 'local', model_path: model.model_path }
}

function InferenceView({ models, model, onModelChange }: { models: WorkbenchModel[]; model: ModelSelection; onModelChange: (value: ModelSelection) => void }) {
  const { setGridRef, widths, ...resizeHandlers } = useResizableSidebars()
  const [session, setSession] = useState<InferenceSession | null>(null)
  const [currentId, setCurrentId] = useState('')
  const [conf, setConf] = useState(0.25)
  const [imgsz, setImgsz] = useState(640)
  const [autoImgsz, setAutoImgsz] = useState(true)
  const [visible, setVisible] = useState<Set<number>>(new Set())
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const fileRef = useRef<HTMLInputElement | null>(null)
  const roiSaveTimer = useRef<number | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      const saved = sessionStorage.getItem('yolo_workbench_session')
      let result: InferenceSession
      try {
        result = (saved ? await api.getWorkbenchSession(saved) : await api.createWorkbenchSession()) as InferenceSession
      } catch {
        sessionStorage.removeItem('yolo_workbench_session')
        result = await api.createWorkbenchSession() as InferenceSession
      }
      if (cancelled) return
      setSession(result)
      setVisible(new Set((result.classes || []).map((item) => item.class_id)))
      sessionStorage.setItem('yolo_workbench_session', result.session_id)
      if (result.images?.length) setCurrentId(result.images[0].image_id)
    }
    void load()
    return () => {
      cancelled = true
      if (roiSaveTimer.current !== null) window.clearTimeout(roiSaveTimer.current)
    }
  }, [])

  const upload = async (files: File[]) => {
    if (!files.length || !session) return
    setBusy(true); setError('')
    try {
      const added = await api.uploadWorkbenchImages(session.session_id, files)
      const refreshed = await api.getWorkbenchSession(session.session_id) as InferenceSession
      setSession(refreshed)
      if (!currentId && added.images?.length) setCurrentId(added.images[0].image_id)
      if (added.rejected?.length) setError(`${added.rejected.length} 个文件未导入：${added.rejected.slice(0, 3).map((item) => item.name).join('、')}`)
    } catch (err) { setError(errorMessage(err, '导入图片失败')) } finally { setBusy(false) }
  }

  const infer = async (all: boolean) => {
    if (!session || (!all && !currentId)) return
    setBusy(true); setError('')
    try {
      const job = await api.inferWorkbench(session.session_id, {
        ...modelPayload(model), conf, imgsz: autoImgsz ? null : imgsz, image_ids: all ? undefined : [currentId],
        rois: Object.fromEntries((session.images || []).map((item) => [item.image_id, item.roi || null])),
      })
      const completed = await waitForJob<InferenceSession>(job.job_id)
      if (completed.result) {
        setSession(completed.result)
        setVisible(new Set((completed.result.classes || []).map((item) => item.class_id)))
        const failed = (completed.result.images || []).find((item) => item.status === 'failed' && item.error)
        if (failed) setError(`${failed.name}: ${failed.error}`)
      }
      setSession(await api.getWorkbenchSession(session.session_id) as InferenceSession)
    } catch (err) { setError(errorMessage(err, '推理失败')) } finally { setBusy(false) }
  }

  const deleteSelected = async () => {
    if (!session || !selectedIds.size) return
    if (!window.confirm(`删除选中的 ${selectedIds.size} 张缓存图片？`)) return
    setBusy(true); setError('')
    try {
      const result = await api.deleteWorkbenchImages(session.session_id, Array.from(selectedIds)) as InferenceSession
      setSession(result)
      setSelectedIds(new Set())
      if (selectedIds.has(currentId)) setCurrentId(result.images?.[0]?.image_id || '')
    } catch (err) { setError(errorMessage(err, '删除图片失败')) } finally { setBusy(false) }
  }

  const updateRoi = (nextRoi: WorkbenchRoi | null) => {
    if (!session || !current) return
    setSession((previous) => previous ? {
      ...previous,
      images: previous.images.map((item) => item.image_id === current.image_id
        ? { ...item, roi: nextRoi, detections: [], status: 'pending', error: '' }
        : item),
    } : previous)
    if (roiSaveTimer.current !== null) window.clearTimeout(roiSaveTimer.current)
    roiSaveTimer.current = window.setTimeout(() => {
      void api.updateWorkbenchImageRoi(session.session_id, current.image_id, nextRoi)
        .catch((err) => setError(errorMessage(err, '保存 ROI 失败')))
    }, 250)
  }

  const rotateCurrent = async (direction: 'clockwise' | 'counterclockwise') => {
    if (!session || !current) return
    if (roiSaveTimer.current !== null) window.clearTimeout(roiSaveTimer.current)
    setBusy(true); setError('')
    try {
      setSession(await api.rotateWorkbenchImage(session.session_id, current.image_id, direction) as InferenceSession)
    } catch (err) { setError(errorMessage(err, '旋转图片失败')) } finally { setBusy(false) }
  }

  const images: WorkbenchImage[] = session?.images || []
  const modelReady = model.model_source === 'platform' ? Boolean(model.trial_id && model.checkpoint_name) : Boolean(model.model_path.trim())
  const current = images.find((item) => item.image_id === currentId) || images[0]
  const sessionId = session?.session_id || ''
  const url = (item: WorkbenchImage) => `/api/workbench/sessions/${sessionId}/images/${item.image_id}/file?v=${item.revision || 0}`
  return (
    <div className="workbench-body">
      <div className="workbench-toolbar">
        <ModelSelector models={models} value={model} disabled={busy} onChange={onModelChange} />
        <label className="compact-field">conf<input className="input" type="number" min="0.001" max="1" step="0.01" value={conf} onChange={(event) => setConf(Number(event.target.value))} /></label>
        <label className="compact-field">imgsz<input className="input" type="number" min="32" max="4096" step="32" value={imgsz} disabled={autoImgsz} onChange={(event) => setImgsz(Number(event.target.value))} /></label>
        <label className="switch-field" title="使用模型 checkpoint 中保存的 imgsz"><input type="checkbox" checked={autoImgsz} onChange={(event) => setAutoImgsz(event.target.checked)} />自动</label>
        <input ref={fileRef} hidden type="file" accept="image/*" multiple onChange={(event) => { void upload(Array.from(event.target.files || [])); event.target.value = '' }} />
        <button className="btn" disabled={busy} onClick={() => fileRef.current?.click()}><ImagePlus size={16} /> 导入图片</button>
        <button className="btn btn-primary" disabled={busy || !current || !modelReady} onClick={() => void infer(false)}>{busy ? <Loader2 className="spin" size={16} /> : <Play size={16} />} 推理</button>
        <button className="btn" disabled={busy || !images.length || !modelReady} onClick={() => void infer(true)}><PlayCircle size={16} /> 全部推理</button>
        {error && <span className="toolbar-error">{error}</span>}
      </div>
      <div ref={setGridRef} className="workbench-main-grid" style={{ '--image-rail-width': `${widths.left}px`, '--class-panel-width': `${widths.right}px` } as React.CSSProperties}>
        <ImageRail images={images} currentId={current?.image_id || ''} imageUrl={url} onSelect={setCurrentId} selectedIds={selectedIds} onSelectionChange={setSelectedIds} onDeleteSelected={() => void deleteSelected()} deleteDisabled={busy} />
        <main className="viewer-region">
          {current ? <OverlayViewer imageUrl={url(current)} imageName={current.name} width={current.width} height={current.height} layers={[{ title: current.roi ? 'Original + ROI Predict' : 'Original + Predict', boxes: current.detections || [] }]} visibleClasses={visible} showResults roi={current.roi} onRoiChange={updateRoi} onRotateLeft={() => void rotateCurrent('counterclockwise')} onRotateRight={() => void rotateCurrent('clockwise')} controlsDisabled={busy} /> : <EmptyState icon={<ImagePlus size={34} />} text="导入图片后开始推理" />}
        </main>
        <ClassFilter classes={session?.classes || []} boxes={current?.detections || []} visible={visible} onChange={setVisible} />
        <SidebarResizeHandles widths={widths} {...resizeHandlers} />
      </div>
    </div>
  )
}

function EvaluationView({ models, model, onModelChange, initialDatasetPath, initialImgsz }: { models: WorkbenchModel[]; model: ModelSelection; onModelChange: (value: ModelSelection) => void; initialDatasetPath: string; initialImgsz: number }) {
  const { setGridRef, widths, ...resizeHandlers } = useResizableSidebars()
  const [datasetPath, setDatasetPath] = useState(initialDatasetPath)
  const [inspection, setInspection] = useState<DatasetInspection | null>(null)
  const [result, setResult] = useState<EvaluationResult | null>(null)
  const [currentId, setCurrentId] = useState('')
  const [conf, setConf] = useState(0.25)
  const [imgsz, setImgsz] = useState(initialImgsz)
  const [batch, setBatch] = useState(8)
  const [visible, setVisible] = useState<Set<number>>(new Set())
  const [metricsExpanded, setMetricsExpanded] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const inspect = async () => {
    setBusy(true); setError('')
    try { setInspection(await api.inspectWorkbenchDataset(datasetPath)) }
    catch (err) { setInspection(null); setError(errorMessage(err, '数据集检查失败')) }
    finally { setBusy(false) }
  }
  const evaluate = async () => {
    setBusy(true); setError(''); setResult(null)
    try {
      if (!inspection) setInspection(await api.inspectWorkbenchDataset(datasetPath))
      const job = await api.evaluateWorkbench({ ...modelPayload(model), dataset_path: datasetPath, conf, imgsz, batch })
      const completed = await waitForJob<EvaluationResult>(job.job_id)
      if (!completed.result) throw new Error('评估任务没有返回结果')
      setResult(completed.result)
      setCurrentId(completed.result.images?.[0]?.image_id || '')
      setVisible(new Set((completed.result.classes || []).map((item) => item.class_id)))
    } catch (err) { setError(errorMessage(err, '评估失败')) } finally { setBusy(false) }
  }
  const images: WorkbenchImage[] = result?.images || []
  const modelReady = model.model_source === 'platform' ? Boolean(model.trial_id && model.checkpoint_name) : Boolean(model.model_path.trim())
  const current = images.find((item) => item.image_id === currentId) || images[0]
  const evaluationId = result?.evaluation_id || ''
  const url = (item: WorkbenchImage) => `/api/workbench/evaluations/${evaluationId}/images/${item.image_id}/file`
  const allBoxes = [...(current?.labels || []), ...(current?.detections || [])]
  return (
    <div className={`workbench-body evaluation-body ${result ? 'has-metrics' : ''}`}>
      <div className="workbench-toolbar evaluation-toolbar">
        <ModelSelector models={models} value={model} disabled={busy} onChange={onModelChange} />
        <label className="dataset-field">验证集路径<input className="input" value={datasetPath} placeholder="data.yaml 或图片与标注所在目录" onChange={(event) => { setDatasetPath(event.target.value); setInspection(null) }} /></label>
        <label className="compact-field">结果 conf<input className="input" type="number" min="0.001" max="1" step="0.01" value={conf} onChange={(event) => setConf(Number(event.target.value))} /></label>
        <label className="compact-field">imgsz<input className="input" type="number" min="32" step="32" value={imgsz} onChange={(event) => setImgsz(Number(event.target.value))} /></label>
        <label className="compact-field">batch<input className="input" type="number" min="1" value={batch} onChange={(event) => setBatch(Number(event.target.value))} /></label>
        <button className="btn" disabled={busy || !datasetPath} onClick={() => void inspect()}><FolderOpen size={16} /> 检查</button>
        <button className="btn btn-primary" disabled={busy || !datasetPath || !modelReady} onClick={() => void evaluate()}>{busy ? <Loader2 className="spin" size={16} /> : <ActivitySquare size={16} />} 开始评估</button>
        {inspection && <span className="dataset-ok"><CheckCircle2 size={15} /> {inspection.dataset_type.toUpperCase()} · {inspection.image_count} 张</span>}
        {error && <span className="toolbar-error">{error}</span>}
      </div>
      {result && <MetricsBand result={result} expanded={metricsExpanded} onToggle={() => setMetricsExpanded((current) => !current)} />}
      <div ref={setGridRef} className="workbench-main-grid" style={{ '--image-rail-width': `${widths.left}px`, '--class-panel-width': `${widths.right}px` } as React.CSSProperties}>
        <ImageRail images={images} currentId={current?.image_id || ''} imageUrl={url} onSelect={setCurrentId} />
        <main className="viewer-region">
          {current ? <OverlayViewer imageUrl={url(current)} imageName={current.name} width={current.width} height={current.height} layers={[{ title: 'Label', boxes: current.labels || [], color: '#22c55e' }, { title: 'Predict', boxes: current.detections || [], color: '#f97316' }]} visibleClasses={visible} showResults /> : <EmptyState icon={<ScanSearch size={34} />} text="选择模型和验证集后开始评估" />}
        </main>
        <ClassFilter classes={result?.classes || []} boxes={allBoxes} visible={visible} onChange={setVisible} />
        <SidebarResizeHandles widths={widths} {...resizeHandlers} />
      </div>
    </div>
  )
}

function MetricsBand({ result, expanded, onToggle }: { result: EvaluationResult; expanded: boolean; onToggle: () => void }) {
  const labels: Record<string, string> = { map50: 'mAP50', map50_95: 'mAP50-95', precision: 'Precision', recall: 'Recall' }
  return (
    <section className={`metrics-band ${expanded ? '' : 'collapsed'}`}>
      <header className="metrics-band-header">
        <strong>评估指标</strong>
        <button className="icon-btn" title={expanded ? '收起评估指标' : '展开评估指标'} onClick={onToggle}>
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </header>
      {expanded && <>
        <div className="metric-summary">
          {Object.entries(labels).map(([key, label]) => <div key={key}><span>{label}</span><strong>{typeof result.metrics?.[key] === 'number' ? result.metrics[key].toFixed(4) : '-'}</strong></div>)}
          <p title={result.predictions_dir}>XML：{result.predictions_dir}</p>
        </div>
        <div className="metric-table-wrap">
          <table><thead><tr><th>类别</th><th>mAP50</th><th>mAP50-95</th><th>Precision</th><th>Recall</th></tr></thead>
            <tbody>{(result.per_class_metrics || []).map((row) => <tr key={row.class_id}><td>{row.class_name}</td><td>{formatMetric(row.map50)}</td><td>{formatMetric(row.map50_95)}</td><td>{formatMetric(row.precision)}</td><td>{formatMetric(row.recall)}</td></tr>)}</tbody>
          </table>
        </div>
      </>}
    </section>
  )
}

function EmptyState({ icon, text }: { icon: React.ReactNode; text: string }) {
  return <div className="workbench-empty">{icon}<span>{text}</span></div>
}

const formatMetric = (value: unknown) => typeof value === 'number' ? value.toFixed(4) : '-'

async function waitForJob<T>(jobId: string): Promise<JobResult<T>> {
  while (true) {
    const job = await api.getJob(jobId) as JobResult<T>
    if (job.status === 'completed') return job
    if (job.status === 'failed') throw new Error(job.error || '任务失败')
    await new Promise((resolve) => window.setTimeout(resolve, 1000))
  }
}
