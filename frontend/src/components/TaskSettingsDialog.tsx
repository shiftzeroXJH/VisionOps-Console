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

  return <div style={{ position: 'fixed', inset: 0, zIndex: 120, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.45)' }} onClick={onClose} /><div className="card flex-col gap-3" style={{ position: 'relative', width: 720, maxWidth: '96vw', maxHeight: '92vh', overflow: 'auto' }}>
    <div className="flex justify-between items-center"><h2 style={{ fontSize: '1.2rem' }}>任务设置</h2><button className="btn" onClick={onClose}>关闭</button></div>
    {error && <div className="text-danger p-2">{error}</div>}
    <label>任务名称<input className="input" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} /></label>
    <label>本地数据集目录<input className="input" value={form.dataset_root} onChange={(e) => setForm({ ...form, dataset_root: e.target.value })} /></label>
    <label>本地数据集 YAML<input className="input" value={form.dataset_yaml} onChange={(e) => setForm({ ...form, dataset_yaml: e.target.value })} /></label>
    <label>本地默认模型<input className="input" value={form.pretrained_model} onChange={(e) => setForm({ ...form, pretrained_model: e.target.value })} /></label>
    <label>服务器<select className="input" value={serverId} onChange={(e) => setServerId(e.target.value)}><option value="">选择服务器</option>{servers.map((s) => <option key={s.remote_server_id} value={s.remote_server_id}>{s.name} ({s.host})</option>)}</select></label>
    {serverId && <><label>远程数据集目录<input className="input" value={form.remote_dataset_root} onChange={(e) => setForm({ ...form, remote_dataset_root: e.target.value })} /></label><label>远程数据集 YAML<input className="input" value={form.remote_dataset_yaml} onChange={(e) => setForm({ ...form, remote_dataset_yaml: e.target.value })} /></label><label>远程默认模型<input className="input" value={form.remote_pretrained_model} onChange={(e) => setForm({ ...form, remote_pretrained_model: e.target.value })} /></label></>}
    <div className="flex justify-end gap-2"><button className="btn" onClick={onClose}>取消</button><button className="btn btn-primary" onClick={() => void save()} disabled={busy}>{busy ? '保存中...' : '保存设置'}</button></div>
  </div></div>
}
