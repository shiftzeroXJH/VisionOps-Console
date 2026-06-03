import { useState } from 'react'
import { api } from '../api'

interface Props {
  experimentId: string
  onClose: () => void
  onImported: () => void
}

export function LocalTrialDialog({ experimentId, onClose, onImported }: Props) {
  const [runDir, setRunDir] = useState('')
  const [pretrained, setPretrained] = useState('')
  const [note, setNote] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async () => {
    setLoading(true)
    setError('')
    try {
      await api.importTrial(experimentId, {
        run_dir: runDir,
        pretrained: pretrained || undefined,
        note: note || undefined,
      })
      onImported()
      onClose()
    } catch (err: any) {
      setError(err?.detail?.error || '导入本地训练失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.45)' }} onClick={onClose} />
      <div className="card flex-col gap-4" style={{ position: 'relative', width: 640, maxWidth: '96vw' }}>
        <div className="flex justify-between items-center">
          <h2 style={{ fontSize: '1.25rem' }}>导入本地训练</h2>
          <button className="btn" onClick={onClose}>关闭</button>
        </div>

        {error && <div className="text-danger p-2" style={{ background: 'rgba(239,68,68,0.1)', borderRadius: 4 }}>{error}</div>}

        <div className="flex-col gap-2">
          <label>本地 run 目录</label>
          <input className="input" value={runDir} onChange={(event) => setRunDir(event.target.value)} placeholder="D:/project/yolo_platform/runs/detect/train42" />
        </div>

        <div className="flex-col gap-2">
          <label>模型路径（可选）</label>
          <input className="input" value={pretrained} onChange={(event) => setPretrained(event.target.value)} placeholder="留空则优先读取 args.yaml 中的 model" />
        </div>

        <div className="flex-col gap-2">
          <label>备注（可选）</label>
          <input className="input" value={note} onChange={(event) => setNote(event.target.value)} placeholder="例如：外部训练结果导入" />
        </div>

        <div className="text-muted" style={{ fontSize: '0.8rem' }}>
          目录内至少需要有 `results.csv`。如果存在 `args.yaml`，系统会优先从中推断模型和训练参数。
        </div>

        <div className="flex justify-end gap-2">
          <button className="btn" onClick={onClose}>取消</button>
          <button className="btn btn-primary" onClick={submit} disabled={loading || !runDir.trim()}>
            {loading ? '导入中...' : '确认导入'}
          </button>
        </div>
      </div>
    </div>
  )
}


