import { useEffect, useState } from 'react'
import { api } from '../api'

type Props = { experimentId: string; onClose: () => void; onSaved: () => void }

export function TaskSettingsDialog({ experimentId, onClose, onSaved }: Props) {
  const [detail, setDetail] = useState<any>(null)
  const [servers, setServers] = useState<any[]>([])
  const [serverId, setServerId] = useState('')
  const [form, setForm] = useState({ description: '', dataset_root: '', dataset_yaml: '', pretrained_model: '', remote_dataset_root: '', remote_dataset_yaml: '', remote_pretrained_model: '' })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([api.getExperiment(experimentId), api.getRemoteServers()]).then(([data, serverData]) => {
      setDetail(data); setServers(serverData.remote_servers || [])
      const exp = data.experiment
      setForm((current) => ({ ...current, description: exp.description || '', dataset_root: exp.dataset_root || '', dataset_yaml: exp.dataset_yaml || '', pretrained_model: exp.pretrained_model || '' }))
      if (serverData.remote_servers?.[0]) setServerId(serverData.remote_servers[0].remote_server_id)
    }).catch((err) => setError(err?.detail?.error || '加载任务设置失败'))
  }, [experimentId])

  useEffect(() => {
    if (!serverId || !detail?.experiment) return
    const remote = detail.experiment.remote_configs?.[serverId] || {}
    setForm((current) => ({ ...current, remote_dataset_root: remote.dataset_root || '', remote_dataset_yaml: remote.dataset_yaml || '', remote_pretrained_model: remote.pretrained_model || '' }))
  }, [serverId, detail])

  const save = async () => {
    setBusy(true); setError('')
    try {
      const remoteConfigs = { ...(detail?.experiment?.remote_configs || {}) }
      if (serverId) remoteConfigs[serverId] = { dataset_root: form.remote_dataset_root.trim(), dataset_yaml: form.remote_dataset_yaml.trim(), pretrained_model: form.remote_pretrained_model.trim() }
      await api.updateExperiment(experimentId, { description: form.description.trim(), dataset_root: form.dataset_root.trim(), dataset_yaml: form.dataset_yaml.trim(), pretrained_model: form.pretrained_model.trim(), remote_configs: remoteConfigs })
      onSaved(); onClose()
    } catch (err: any) { setError(err?.detail?.error || '保存任务设置失败') } finally { setBusy(false) }
  }

  return (
    <div className="dialog-overlay" onClick={(e) => { if (e.target === e.currentTarget && !busy) onClose() }}>
      <div className="card dialog-card flex-col gap-3" style={{ width: 640, maxHeight: '92vh', overflowY: 'auto' }}>
        <div className="flex justify-between items-center" style={{ borderBottom: '1px solid var(--border-default)', paddingBottom: '0.75rem' }}>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-primary)' }}>任务设置</h2>
          <button className="btn" style={{ padding: '0.25rem 0.5rem' }} onClick={onClose}>关闭</button>
        </div>
        {error && <div className="text-danger p-2">{error}</div>}
        <div className="flex-col gap-3 pt-2">
          <label style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-regular)' }}>任务名称<input className="input mt-1" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} /></label>
          <label style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-regular)' }}>本地数据集目录<input className="input mt-1 font-mono" value={form.dataset_root} onChange={(e) => setForm({ ...form, dataset_root: e.target.value })} /></label>
          <label style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-regular)' }}>本地数据集 YAML<input className="input mt-1 font-mono" value={form.dataset_yaml} onChange={(e) => setForm({ ...form, dataset_yaml: e.target.value })} /></label>
          <label style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-regular)' }}>本地默认模型<input className="input mt-1 font-mono" value={form.pretrained_model} onChange={(e) => setForm({ ...form, pretrained_model: e.target.value })} /></label>
          <label style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-regular)' }}>服务器<select className="input mt-1" value={serverId} onChange={(e) => setServerId(e.target.value)}><option value="">选择服务器</option>{servers.map((s) => <option key={s.remote_server_id} value={s.remote_server_id}>{s.name} ({s.host})</option>)}</select></label>
          {serverId && (
            <div className="flex-col gap-3 p-3 mt-1" style={{ borderRadius: 'var(--radius-sm)', background: 'var(--bg-subtle)', border: '1px solid var(--border-default)' }}>
              <label style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-regular)' }}>远程数据集目录<input className="input mt-1 font-mono" value={form.remote_dataset_root} onChange={(e) => setForm({ ...form, remote_dataset_root: e.target.value })} /></label>
              <label style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-regular)' }}>远程数据集 YAML<input className="input mt-1 font-mono" value={form.remote_dataset_yaml} onChange={(e) => setForm({ ...form, remote_dataset_yaml: e.target.value })} /></label>
              <label style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-regular)' }}>远程默认模型<input className="input mt-1 font-mono" value={form.remote_pretrained_model} onChange={(e) => setForm({ ...form, remote_pretrained_model: e.target.value })} /></label>
            </div>
          )}
        </div>
        <div className="flex justify-end gap-2 pt-3" style={{ borderTop: '1px solid var(--border-default)' }}>
          <button className="btn" onClick={onClose}>取消</button>
          <button className="btn btn-primary" onClick={() => void save()} disabled={busy}>{busy ? '保存中...' : '保存设置'}</button>
        </div>
      </div>
    </div>
  )
}
