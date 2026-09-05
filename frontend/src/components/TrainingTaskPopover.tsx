import { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowDown, ArrowUp, ChevronDown, ChevronRight, ListTodo, RefreshCw, Trash2, X } from 'lucide-react'
import { api, type TrainingTask, type TrainingTaskList } from '../api'

interface Props {
  data: TrainingTaskList
  onClose: () => void
  onChanged: () => Promise<void>
  onSelectExperiment: (experimentId: string) => void
}

const formatDateTime = (value?: string) => {
  if (!value) return '-'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

const modelName = (value?: string | null) => value?.replace(/\\/g, '/').split('/').pop() || '模型信息不可用'

const errorMessage = (error: unknown, fallback: string) => {
  if (!error || typeof error !== 'object' || !("detail" in error)) return fallback
  const detail = error.detail
  if (!detail || typeof detail !== 'object' || !("error" in detail)) return fallback
  return typeof detail.error === 'string' ? detail.error : fallback
}

function TaskRow({ task, queued, index, count, busy, run, onSelectExperiment }: {
  task: TrainingTask
  queued: boolean
  index: number
  count: number
  busy: boolean
  run: (action: () => Promise<unknown>) => Promise<void>
  onSelectExperiment: (experimentId: string) => void
}) {
  const phase = task.phase || 'running'
  return <div className="training-task-row">
    <button className="training-task-main" disabled={!task.experiment_id} onClick={() => onSelectExperiment(task.experiment_id)}>
      <span className={`status-pill ${queued ? 'status-pill-queued' : phase === 'running' ? 'status-pill-running' : 'status-pill-neutral'}`} style={{ alignSelf: 'center' }}>
        <span className="status-dot" />
        {queued ? `#${index + 1} 排队` : phase === 'preparing' ? '准备中' : phase === 'unknown' ? '未知' : '训练中'}
      </span>
      <strong style={{ fontSize: '0.84rem' }}>{task.experiment_name || task.experiment_id || task.queue_id}</strong>
      <span className="text-muted" style={{ fontSize: '0.72rem' }}>{task.project} · {task.training_mode === 'continued' ? `续训自 ${task.parent_display_name || task.parent_trial_id}` : modelName(task.model)}</span>
      <span className="text-muted font-mono" style={{ fontSize: '0.72rem' }}>{queued ? `提交：${formatDateTime(task.created_at)}` : `开始：${formatDateTime(task.started_at)}`}</span>
      {(task.last_synced_epoch_count ?? 0) > 0 && <span className="training-task-detail font-mono" style={{ color: 'var(--primary)' }}>已同步 {task.last_synced_epoch_count} Epoch</span>}
      {task.waiting_reason && <span className="training-task-detail">{task.waiting_reason}</span>}
      {task.error && <span className="training-task-detail text-danger">{task.error}</span>}
    </button>
    <div className="training-task-actions">
      {queued && <>
        <button className="btn" style={{ padding: '0.2rem 0.35rem', height: 26 }} disabled={busy || index === 0} onClick={() => void run(() => api.reorderTrainingTask(task.queue_id, index))} title="上移" aria-label="上移"><ArrowUp size={13} /></button>
        <button className="btn" style={{ padding: '0.2rem 0.35rem', height: 26 }} disabled={busy || index === count - 1} onClick={() => void run(() => api.reorderTrainingTask(task.queue_id, index + 2))} title="下移" aria-label="下移"><ArrowDown size={13} /></button>
        <button className="btn btn-danger" style={{ padding: '0.2rem 0.35rem', height: 26 }} disabled={busy} onClick={() => {
          if (confirm(`确定取消“${task.experiment_name}”的排队训练吗？`)) void run(() => api.cancelTrainingTask(task.queue_id))
        }} title="取消排队" aria-label="取消排队"><Trash2 size={13} /></button>
      </>}
      {!queued && phase === 'unknown' && <button className="btn" style={{ padding: '0.2rem 0.5rem', height: 26, fontSize: '0.75rem' }} disabled={busy} onClick={() => void run(() => api.recheckTrainingTask(task.queue_id))}>重新检查</button>}
    </div>
  </div>
}

const EXPANSION_KEY = 'training-task-groups-expanded'

export function TrainingTaskPopover({ data, onChanged, onSelectExperiment, onClose }: Props) {
  const root = useRef<HTMLDivElement>(null)
  const busyRef = useRef(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState<Record<string, boolean>>(() => {
    try {
      const saved: unknown = JSON.parse(localStorage.getItem(EXPANSION_KEY) || '{}')
      return saved && typeof saved === 'object' ? Object.fromEntries(Object.entries(saved).filter(([, value]) => typeof value === 'boolean')) : {}
    } catch { return {} }
  })
  const groups = useMemo(() => (data.groups ?? [{
    target_id: 'local', name: '本地', source: 'local' as const, remote_server_id: null,
    max_parallel_training_tasks: data.max_parallel_training_tasks,
    running_count: data.running_count, queued_count: data.queued_count,
    running: data.running, queued: data.queued, blocked: false, last_failure: null,
  }]).slice().sort((a, b) => Number(b.source === 'local') - Number(a.source === 'local')), [data])

  useEffect(() => {
    setExpanded((previous) => {
      const next = { ...previous }
      let changed = false
      for (const group of groups) {
        if (!(group.target_id in next)) {
          next[group.target_id] = group.running_count > 0 || group.queued_count > 0 || group.blocked || !!group.last_failure
          changed = true
        }
      }
      return changed ? next : previous
    })
  }, [groups])

  useEffect(() => {
    try { localStorage.setItem(EXPANSION_KEY, JSON.stringify(expanded)) } catch { /* Storage may be unavailable. */ }
  }, [expanded])

  useEffect(() => {
    const outside = (event: PointerEvent) => {
      if (event.target instanceof Element && !root.current?.contains(event.target) && !event.target.closest('.training-queue-trigger')) onClose()
    }
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    document.addEventListener('pointerdown', outside)
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('pointerdown', outside)
      document.removeEventListener('keydown', escape)
    }
  }, [onClose])

  const run = async (action: () => Promise<unknown>) => {
    if (busyRef.current) return
    busyRef.current = true
    setBusy(true); setError('')
    try { await action(); await onChanged() }
    catch (err) { setError(errorMessage(err, '队列操作失败，请刷新后重试')) }
    finally { busyRef.current = false; setBusy(false) }
  }

  return <div className="training-task-popover" ref={root} role="dialog" aria-label="模型训练列表" aria-busy={busy}>
    <div className="training-task-popover-title">
      <ListTodo size={16} /><strong>模型训练列表</strong>
      <span className="training-task-totals">运行 {data.total_running_count ?? data.running_count} · 排队 {data.total_queued_count ?? data.queued_count}</span>
      <button className="icon-btn" disabled={busy} onClick={() => void run(async () => {})} title="刷新" aria-label="刷新"><RefreshCw size={16} /></button>
      <button className="icon-btn" onClick={onClose} title="关闭" aria-label="关闭"><X size={16} /></button>
    </div>
    <div className="training-task-popover-scroll">
      {error && <div role="alert" className="text-danger">{error}</div>}
      {groups.map((group) => {
        const open = expanded[group.target_id] ?? (group.running_count > 0 || group.queued_count > 0 || group.blocked || !!group.last_failure)
        return <section className="training-task-section" key={group.target_id}>
          <button className="training-task-group-toggle" aria-expanded={open} onClick={() => setExpanded((previous) => ({ ...previous, [group.target_id]: !open }))}>
            {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            <strong>{group.source === 'local' ? '本地' : group.name}</strong>
            <span>运行 {group.running_count}/{group.max_parallel_training_tasks} · 排队 {group.queued_count}</span>
          </button>
          {open && <>
            {group.blocked && <div className="training-task-detail text-warning">队列已阻塞，请检查未知状态任务或最近失败原因。</div>}
            {group.running.map((task, index) => <TaskRow key={task.queue_id} task={task} queued={false} index={index} count={group.running.length} busy={busy} run={run} onSelectExperiment={onSelectExperiment} />)}
            {group.queued.map((task, index) => <TaskRow key={task.queue_id} task={task} queued index={index} count={group.queued.length} busy={busy} run={run} onSelectExperiment={onSelectExperiment} />)}
            {group.last_failure && <div className="training-task-failure"><strong>最近失败：{group.last_failure.experiment_name}</strong><div>{group.last_failure.error || group.last_failure.waiting_reason || '训练失败'}</div></div>}
            {!group.running.length && !group.queued.length && <div className="training-task-empty">当前没有训练或排队任务</div>}
          </>}
        </section>
      })}
    </div>
  </div>
}
