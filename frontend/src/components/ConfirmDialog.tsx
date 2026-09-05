import { useCallback, useEffect, useState } from 'react'

interface Props {
  title: string
  message: string
  confirmLabel?: string
  confirmClassName?: string
  onConfirm: () => Promise<void>
  onClose: () => void
}

export function ConfirmDialog({
  title,
  message,
  confirmLabel = '确认',
  confirmClassName = 'btn btn-primary',
  onConfirm,
  onClose,
}: Props) {
  const [loading, setLoading] = useState(false)

  const handleConfirm = async () => {
    setLoading(true)
    try {
      await onConfirm()
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
      <div className="card dialog-card" style={{ width: '440px' }}>
        <h2 style={{ marginBottom: '0.75rem', fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-primary)' }}>{title}</h2>
        <p style={{ marginBottom: '1rem', color: 'var(--text-regular)', fontSize: '0.875rem' }}>{message}</p>

        <div className="flex justify-end gap-2 pt-4" style={{ borderTop: '1px solid var(--panel-border)' }}>
          <button className="btn" onClick={onClose} disabled={loading}>取消</button>
          <button className={confirmClassName} onClick={handleConfirm} disabled={loading}>
            {loading ? '处理中...' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
