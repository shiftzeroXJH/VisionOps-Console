import { useState } from 'react'
import { Trash2, X } from 'lucide-react'
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
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')

  const clearValidationCache = async () => {
    if (!confirm('确定清除所有验证缓存吗？这只会删除验证可视化临时图片，不会删除训练记录。')) return
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

        <section className="settings-section">
          <div>
            <h3>验证缓存</h3>
            <p className="text-muted">清除 Trial 验证功能生成的 label/predict 临时图片，避免长时间使用后占用过多磁盘空间。</p>
          </div>
          <button className="btn btn-danger" onClick={clearValidationCache} disabled={clearing}>
            <Trash2 size={16} /> {clearing ? '清除中...' : '清除缓存'}
          </button>
        </section>

        {result && (
          <div className="settings-result">
            已清除 {result.deleted_dirs || 0} 个缓存目录、{result.deleted_files || 0} 个文件，释放 {formatBytes(result.deleted_bytes || 0)}。
            {Array.isArray(result.warnings) && result.warnings.length > 0 && (
              <div className="text-warning" style={{ marginTop: '0.5rem' }}>{result.warnings.join('；')}</div>
            )}
          </div>
        )}
        {error && <div className="settings-result text-danger">{error}</div>}
      </div>
    </div>
  )
}
