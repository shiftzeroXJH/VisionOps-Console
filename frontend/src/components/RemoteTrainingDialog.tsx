import { useEffect, useState } from 'react'
import { api } from '../api'

type Props = {
  experimentId: string
  params: Record<string, unknown>
  pretrained: string
  note: string
  onClose: () => void
  onStarted: () => void
}

export function RemoteTrainingDialog({ experimentId, params, pretrained, note, onClose, onStarted }: Props) {
  const [detail, setDetail] = useState<any>(null)
  const [servers, setServers] = useState<any[]>([])
  const [serverId, setServerId] = useState('')
  const [datasetRoot, setDatasetRoot] = useState('')
  const [datasetYaml, setDatasetYaml] = useState('')
  const [remoteModel, setRemoteModel] = useState(pretrained.split(/[\\/]/).pop() || pretrained)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const load = async () => {
    const [experiment, serverData] = await Promise.all([api.getExperiment(experimentId), api.getRemoteServers()])
    setDetail(experiment)
    const items = serverData.remote_servers || []
    setServers(items)
    if (!serverId && items[0]) setServerId(items[0].remote_server_id)
  }

  useEffect(() => { load().catch((err) => setError(err?.detail?.error || '加载远程配置失败')) }, [])

  useEffect(() => {
    if (!serverId || !detail?.experiment) return
    const cfg = detail.experiment.remote_configs?.[serverId] || {}
    setDatasetRoot(cfg.dataset_root || '')
    setDatasetYaml(cfg.dataset_yaml || '')
    setRemoteModel(cfg.pretrained_model || (pretrained.split(/[\\/]/).pop() || pretrained))
  }, [serverId, detail, pretrained])

  const submit = async () => {
    if (!serverId) return
    setBusy(true); setError('')
    try {
      const configs = { ...(detail?.experiment?.remote_configs || {}), [serverId]: { dataset_root: datasetRoot.trim(), dataset_yaml: datasetYaml.trim(), pretrained_model: remoteModel.trim() } }
      await api.updateExperiment(experimentId, { remote_configs: configs } as any)
      await api.runRemoteTrial(experimentId, { remote_server_id: serverId, params, pretrained: remoteModel.trim(), note })
      onStarted(); onClose()
    } catch (err: any) { setError(err?.detail?.error || '远程训练启动失败') } finally { setBusy(false) }
  }

  return <div style={{ position: 'fixed', inset: 0, zIndex: 120, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.45)' }} onClick={onClose} />
    <div className="card flex-col gap-3" style={{ position: 'relative', width: 760, maxWidth: '96vw', maxHeight: '92vh', overflow: 'auto' }}>
      <div className="flex justify-between items-center"><h2 style={{ fontSize: '1.2rem' }}>远程训练</h2><button className="btn" onClick={onClose}>关闭</button></div>
      {error && <div className="text-danger p-2">{error}</div>}
      <label>远程服务器<select className="input" value={serverId} onChange={(e) => setServerId(e.target.value)}><option value="" disabled>请选择已配置服务器</option>{servers.map((s) => <option key={s.remote_server_id} value={s.remote_server_id}>{s.name} ({s.username}@{s.host}:{s.port})</option>)}</select></label>
      <label>远程数据集目录<input className="input" value={datasetRoot} onChange={(e) => setDatasetRoot(e.target.value)} placeholder="/media/acp/shuju/XJH/datasets/coc-plate1" /></label>
      <label>远程数据集 YAML（可留空自动查找）<input className="input" value={datasetYaml} onChange={(e) => setDatasetYaml(e.target.value)} placeholder=".../data.yaml" /></label>
      <label>远程默认模型<input className="input" value={remoteModel} onChange={(e) => setRemoteModel(e.target.value)} /></label>
      <div className="text-muted" style={{ fontSize: 12 }}>AMP 检查权重会自动上传到每个 Trial 目录。密码保存在本地配置中，暂不回显。</div>
      <div className="flex justify-end gap-2"><button className="btn" onClick={onClose}>取消</button><button className="btn btn-primary" onClick={() => void submit()} disabled={busy || !serverId || (!datasetRoot && !datasetYaml) || !remoteModel}>{busy ? '启动中...' : '确认启动'}</button></div>
    </div>
  </div>
}
