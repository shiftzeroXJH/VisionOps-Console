import { useEffect, useState } from 'react'
import { Play, X } from 'lucide-react'
import { api } from '../api'

interface Props {
  trialId: string
  onClose: () => void
  onSubmitted: () => void | Promise<void>
}

type ApiError = {
  detail?: {
    error?: string
    code?: string
    running_count?: number
    max_parallel_training_tasks?: number
  }
}

type ContinuationOptions = {
  display_name: string
  can_continue: boolean
  unavailable_reason: string
  cumulative_epochs: number
  defaults: {
    additional_epochs: number
    lr0: number
    original_lr0: number
    patience: number
  }
}

const errorText = (error: unknown, fallback: string) => {
  const detail = (error as ApiError)?.detail
  return detail?.error || (error instanceof Error ? error.message : fallback)
}

export function ContinueTrainingDialog({ trialId, onClose, onSubmitted }: Props) {
  const [options, setOptions] = useState<ContinuationOptions | null>(null)
  const [additionalEpochs, setAdditionalEpochs] = useState(100)
  const [lr0, setLr0] = useState(0.0001)
  const [patience, setPatience] = useState(10)
  const [note, setNote] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api.getTrialContinuation(trialId)
      .then((data) => {
        setOptions(data)
        setAdditionalEpochs(Number(data.defaults?.additional_epochs) || 100)
        setLr0(Number(data.defaults?.lr0) || 0.0001)
        setPatience(Number(data.defaults?.patience) || 0)
      })
      .catch((err: unknown) => setError(errorText(err, '无法读取续训配置')))
  }, [trialId])

  const submit = async () => {
    if (!Number.isInteger(additionalEpochs) || additionalEpochs < 1 || additionalEpochs > 1000) {
      setError('追加 epochs 必须是 1 到 1000 的整数')
      return
    }
    if (!Number.isFinite(lr0) || lr0 < 0.00001 || lr0 > 0.1) {
      setError('初始学习率必须在 0.00001 到 0.1 之间')
      return
    }
    if (!Number.isInteger(patience) || patience < 0 || patience > 300) {
      setError('patience 必须是 0 到 300 的整数')
      return
    }
    const payload = { additional_epochs: additionalEpochs, lr0, patience, note: note.trim() || undefined }
    setLoading(true)
    setError('')
    try {
      try {
        await api.continueTrial(trialId, payload)
      } catch (err: unknown) {
        const detail = (err as ApiError)?.detail
        if (detail?.code !== 'TRAINING_CAPACITY_REACHED') throw err
        const running = Number(detail.running_count) || 0
        const maximum = Number(detail.max_parallel_training_tasks) || 1
        if (!confirm(`当前已有 ${running}/${maximum} 个训练任务运行中，是否加入排队队列？`)) return
        await api.continueTrial(trialId, { ...payload, enqueue_if_busy: true })
      }
      await onSubmitted()
      onClose()
    } catch (err: unknown) {
      setError(errorText(err, '续训任务提交失败'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="dialog-overlay" style={{ zIndex: 80 }} onClick={(event) => { if (event.target === event.currentTarget && !loading) onClose() }}>
      <div className="card continue-training-dialog">
        <div className="flex justify-between items-center">
          <div>
            <h2 style={{ fontSize: '1.15rem' }}>继续训练</h2>
            <div className="text-muted" style={{ fontSize: '0.8rem', marginTop: '0.2rem' }}>
              {options?.display_name || trialId} · 已累计 {options?.cumulative_epochs ?? '-'} epochs
            </div>
          </div>
          <button className="btn icon-btn" onClick={onClose} title="关闭"><X size={18} /></button>
        </div>

        {error && <div className="text-danger p-2" style={{ background: 'rgba(239,68,68,0.1)', borderRadius: 4 }}>{error}</div>}
        {options && !options.can_continue && (
          <div className="text-danger p-2" style={{ background: 'rgba(239,68,68,0.1)', borderRadius: 4 }}>{options.unavailable_reason}</div>
        )}

        <div className="continue-training-grid">
          <label className="param-label">
            <span>追加 epochs</span>
            <input className="input" type="number" min={1} max={1000} step={1} value={additionalEpochs} onChange={(event) => setAdditionalEpochs(Number(event.target.value))} />
          </label>
          <label className="param-label">
            <span>初始学习率 lr0</span>
            <input className="input" type="number" min={0.00001} max={0.1} step="any" value={lr0} onChange={(event) => setLr0(Number(event.target.value))} />
            <span className="param-helper">原值 {options?.defaults?.original_lr0 ?? '-'}，默认使用原值的 10%</span>
          </label>
          <label className="param-label">
            <span>早停 patience</span>
            <input className="input" type="number" min={0} max={300} step={1} value={patience} onChange={(event) => setPatience(Number(event.target.value))} />
          </label>
          <label className="param-label continue-training-note">
            <span>备注</span>
            <input className="input" value={note} onChange={(event) => setNote(event.target.value)} placeholder="可选" />
          </label>
        </div>

        <div className="flex justify-end gap-2">
          <button className="btn" onClick={onClose} disabled={loading}>取消</button>
          <button className="btn btn-primary" onClick={submit} disabled={loading || !options?.can_continue}>
            <Play size={16} /> {loading ? '正在提交...' : '开始续训'}
          </button>
        </div>
      </div>
    </div>
  )
}
