import { useEffect, useState } from 'react'
import { Save, Trash2, X } from 'lucide-react'
import { api } from '../api'

interface Props {
  onClose: () => void
}

const formatBytes = (bytes: number) => {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let unitIndex = 0
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex += 1
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 2)} ${units[unitIndex]}`
}

export function SettingsDialog({ onClose }: Props) {
  const [clearing, setClearing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [loading, setLoading] = useState(true)
  const [yoloPython, setYoloPython] = useState('')
  const [effectivePython, setEffectivePython] = useState('')
  const [maxParallelTrainingTasks, setMaxParallelTrainingTasks] = useState(1)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')
  const [saveMessage, setSaveMessage] = useState('')

  useEffect(() => {
    let cancelled = false

    const loadSettings = async () => {
      setLoading(true)
      setError('')
      try {
        const res = await api.getSettings()
        if (cancelled) return
        setYoloPython(res.yolo_python || '')
        setEffectivePython(res.effective_yolo_python || '')
        setMaxParallelTrainingTasks(Number(res.max_parallel_training_tasks) || 1)
      } catch (err: any) {
        if (cancelled) return
        setError(err?.detail?.error || '加载设置失败')
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void loadSettings()
    return () => {
      cancelled = true
    }
  }, [])

  const clearValidationCache = async () => {
    if (!confirm('确定清除验证与模型工作台缓存吗？这不会删除训练记录或评估输出的预测 XML。')) return
    setClearing(true)
    setError('')
    setResult(null)
    try {
      const res = await api.clearValidationCache()
      setResult(res)
    } catch (err: any) {
      setError(err?.detail?.error || '清除缓存失败')
    } finally {
      setClearing(false)
    }
  }

  const saveSettings = async () => {
    setSaving(true)
    setError('')
    setSaveMessage('')
    try {
      const res = await api.updateSettings({
        yolo_python: yoloPython,
        max_parallel_training_tasks: maxParallelTrainingTasks,
      })
      setYoloPython(res.yolo_python || '')
      setEffectivePython(res.effective_yolo_python || '')
      setMaxParallelTrainingTasks(Number(res.max_parallel_training_tasks) || 1)
      setSaveMessage('已保存。新的并行限制会立即用于训练调度。')
    } catch (err: any) {
      setError(err?.detail?.error || '保存设置失败')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="dialog-overlay">
      <div className="card dialog-card settings-dialog">
        <div className="settings-header">
          <div>
            <div className="settings-kicker">全局设置</div>
            <h2>设置</h2>
          </div>
          <button className="btn" onClick={onClose} title="关闭"><X size={18} /></button>
        </div>

        <section className="settings-section settings-section-stack">
          <div className="settings-form-block">
            <h3>YOLO Python</h3>
            <p className="text-muted">
              设置训练、验证预览、模型推理、模型评估和导出 ONNX 时使用的 Python 可执行文件。留空时会使用当前检测到的默认路径。
            </p>
            <input
              className="input"
              value={yoloPython}
              onChange={(e) => setYoloPython(e.target.value)}
              placeholder="例如：C:\\Users\\Administrator\\miniconda3\\envs\\yolo_env\\python.exe"
              disabled={loading || saving}
            />
            <div className="settings-effective">
              <span className="text-muted">当前生效路径</span>
              <code>{effectivePython || '未检测到'}</code>
            </div>
          </div>
        </section>

        <section className="settings-section settings-section-stack">
          <label className="settings-form-block">
            <h3>最大并行模型训练任务</h3>
            <p className="text-muted">仅限制平台启动的本地调参训练；降低数值不会停止正在运行的任务。</p>
            <input
              className="input"
              type="number"
              min={1}
              max={64}
              step={1}
              value={maxParallelTrainingTasks}
              onChange={(event) => setMaxParallelTrainingTasks(Number(event.target.value))}
              disabled={loading || saving}
            />
          </label>
          <button className="btn btn-primary" onClick={saveSettings} disabled={loading || saving}>
            <Save size={16} /> {saving ? '保存中...' : '保存设置'}
          </button>
        </section>

        <section className="settings-section">
          <div>
            <h3>验证与工作台缓存</h3>
            <p className="text-muted">清除验证预览、上传图片和临时推理结果；不会删除训练记录或预测 XML。</p>
          </div>
          <button className="btn btn-danger" onClick={clearValidationCache} disabled={clearing}>
            <Trash2 size={16} /> {clearing ? '清除中...' : '清除缓存'}
          </button>
        </section>

        {result && (
          <div className="settings-result">
            已清除 {result.deleted_dirs || 0} 个缓存目录，{result.deleted_files || 0} 个文件，释放 {formatBytes(result.deleted_bytes || 0)}。
            {Array.isArray(result.warnings) && result.warnings.length > 0 && (
              <div className="text-warning" style={{ marginTop: '0.5rem' }}>{result.warnings.join('；')}</div>
            )}
          </div>
        )}
        {saveMessage && <div className="settings-result text-success">{saveMessage}</div>}
        {error && <div className="settings-result text-danger">{error}</div>}
      </div>
    </div>
  )
}
