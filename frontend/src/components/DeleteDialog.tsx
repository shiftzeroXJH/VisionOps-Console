import { useCallback, useEffect, useState } from 'react'

interface Props {
  title: string
  message: string
  dangerousMessage?: string
  onConfirm: (keepFiles: boolean) => Promise<void>
  onClose: () => void
}

export function DeleteDialog({ title, message, dangerousMessage, onConfirm, onClose }: Props) {
  const [keepFiles, setKeepFiles] = useState(true)
  const [loading, setLoading] = useState(false)

  const handleConfirm = async () => {
    setLoading(true)
    try {
      await onConfirm(keepFiles)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (event.key === 'Escape' && !loading) onClose()
  }, [loading, onClose])

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  return (
    <div
      className="dialog-overlay"
      onClick={(e) => { if (e.target === e.currentTarget && !loading) onClose() }}
    >
      <div className="card dialog-card" style={{ width: '440px', border: '1px solid #fecaca' }}>
        <h2 style={{ marginBottom: '0.75rem', fontSize: '1.15rem', fontWeight: 700, color: 'var(--danger-color)' }}>{title}</h2>
        <p style={{ marginBottom: '1rem', color: 'var(--text-regular)', fontSize: '0.875rem' }}>{message}</p>

        {dangerousMessage && (
          <div className="p-3 mb-4" style={{ backgroundColor: 'rgba(239,68,68,0.1)', borderRadius: 'var(--radius-sm)' }}>
            <label className="flex items-center gap-2" style={{ cursor: 'pointer', fontSize: '0.875rem' }}>
              <input type="checkbox" checked={!keepFiles} onChange={(e) => setKeepFiles(!e.target.checked)} style={{ width: '1rem', height: '1rem' }} />
              <span className="text-danger" style={{ fontWeight: 600 }}>{dangerousMessage}</span>
            </label>
            {!keepFiles && <div className="text-danger mt-1" style={{ fontSize: '0.75rem' }}>警告：物理文件删除后将无法恢复。</div>}
          </div>
        )}

        <div className="flex justify-end gap-2 pt-4" style={{ borderTop: '1px solid var(--panel-border)' }}>
          <button className="btn" onClick={onClose} disabled={loading}>取消</button>
          <button className="btn btn-danger" onClick={handleConfirm} disabled={loading}>
            {loading ? '正在删除...' : '确认删除'}
          </button>
        </div>
      </div>
    </div>
  )
}
