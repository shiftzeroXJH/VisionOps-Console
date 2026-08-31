import { useEffect, useState } from 'react'
import { Save } from 'lucide-react'
import { api } from '../api'

interface Props {
  trialId: string
  defaultName: string
  onSaved: (name: string, overwritten: boolean) => void
  onClose: () => void
}

type ApiFailure = {
  detail?: {
    code?: string
    error?: string
  }
}

export function SaveHyperparameterTemplateDialog({ trialId, defaultName, onSaved, onClose }: Props) {
  const [name, setName] = useState(defaultName)
  const [saving, setSaving] = useState(false)
  const [confirmOverwrite, setConfirmOverwrite] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !saving) onClose()
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [onClose, saving])

  const save = async (overwrite: boolean) => {
    const normalizedName = name.trim()
    if (!normalizedName) {
      setError('模板名称不能为空')
      return
    }
    if (normalizedName.length > 80) {
      setError('模板名称不能超过 80 个字符')
      return
    }
    setSaving(true)
    setError('')
    try {
      const result = await api.saveTrialHyperparameterTemplate(trialId, {
        name: normalizedName,
        overwrite,
      })
      onSaved(result.template?.name || normalizedName, Boolean(result.overwritten))
    } catch (err: unknown) {
      const failure = err as ApiFailure
      if (failure.detail?.code === 'TEMPLATE_NAME_CONFLICT') {
        setConfirmOverwrite(true)
      } else {
        setError(failure.detail?.error || '保存超参数模板失败')
      }
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="template-dialog-overlay" onClick={(event) => { if (event.target === event.currentTarget && !saving) onClose() }}>
      <div className="card template-dialog">
        <div className="template-dialog-title"><Save size={18} /><h2>保存超参数模板</h2></div>
        {confirmOverwrite ? (
          <p>已存在名为“{name.trim()}”的模板。覆盖后，原模板参数将被当前 Trial 参数替换。</p>
        ) : (
          <>
            <label className="param-label" htmlFor="hyperparameter-template-name">模板名称</label>
            <input
              id="hyperparameter-template-name"
              className="input"
              autoFocus
              maxLength={80}
              value={name}
              onChange={(event) => {
                setName(event.target.value)
                setError('')
              }}
              onKeyDown={(event) => { if (event.key === 'Enter') void save(false) }}
              disabled={saving}
            />
          </>
        )}
        {error && <div className="text-danger" style={{ fontSize: '0.85rem' }}>{error}</div>}
        <div className="flex justify-end gap-2 pt-4 template-dialog-actions">
          <button className="btn" onClick={onClose} disabled={saving}>取消</button>
          <button className="btn btn-primary" onClick={() => void save(confirmOverwrite)} disabled={saving}>
            <Save size={16} /> {saving ? '保存中...' : confirmOverwrite ? '确认覆盖' : '保存'}
          </button>
        </div>
      </div>
    </div>
  )
}
