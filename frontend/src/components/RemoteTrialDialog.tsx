import { useEffect, useState } from 'react'
import { api } from '../api'

interface Props {
  experimentId: string
  onClose: () => void
  onImported: () => void
}

export function RemoteTrialDialog({ experimentId, onClose, onImported }: Props) {
  const [servers, setServers] = useState<any[]>([])
  const [remoteServerId, setRemoteServerId] = useState('')
  const [remoteRunDir, setRemoteRunDir] = useState('')
  const [note, setNote] = useState('')
  const [syncNow, setSyncNow] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const loadServers = async () => {
    const res = await api.getRemoteServers()
    const items = res.remote_servers || []
    setServers(items)
    if (!remoteServerId && items.length > 0) setRemoteServerId(items[0].remote_server_id)
  }

  useEffect(() => {
    loadServers().catch((err) => setError(err?.detail?.error || '加载远程服务器失败'))
  }, [])

  const submit = async () => {
    setLoading(true)
    setError('')
    try {
      if (syncNow) {
        await api.importRemoteTrial(experimentId, {
          remote_server_id: remoteServerId,
          remote_run_dir: remoteRunDir,
          note,
        })
      } else {
        await api.registerRemoteTrial(experimentId, {
          remote_server_id: remoteServerId,
          remote_run_dir: remoteRunDir,
          note,
        })
      }
      onImported()
      onClose()
    } catch (err: any) {
      setError(err?.detail?.error || '登记远程训练失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.45)' }} onClick={onClose} />
      <div className="card flex-col gap-4" style={{ position: 'relative', width: 720, maxWidth: '96vw' }}>
        <div className="flex justify-between items-center">
          <h2 style={{ fontSize: '1.25rem' }}>登记远程训练</h2>
          <button className="btn" onClick={onClose}>关闭</button>
        </div>

        {error && <div className="text-danger p-2" style={{ background: 'rgba(239,68,68,0.1)', borderRadius: 4 }}>{error}</div>}

        <div className="flex-col gap-2">
          <label>远程服务器</label>
          <select className="input" value={remoteServerId} onChange={(e) => setRemoteServerId(e.target.value)}>
            <option value="" disabled>请选择已配置服务器</option>
            {servers.map((server) => (
              <option key={server.remote_server_id} value={server.remote_server_id}>
                {server.name} ({server.username}@{server.host}:{server.port})
              </option>
            ))}
          </select>
        </div>

        <div className="flex-col gap-2">
          <label>远程 run 目录</label>
          <input className="input" value={remoteRunDir} onChange={(e) => setRemoteRunDir(e.target.value)} placeholder="/home/user/runs/detect/train42" />
        </div>
        <div className="flex-col gap-2">
          <label>备注</label>
          <input className="input" value={note} onChange={(e) => setNote(e.target.value)} />
        </div>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={syncNow} onChange={(e) => setSyncNow(e.target.checked)} />
          登记后立即同步一次 `results.csv`
        </label>

        <div className="flex justify-end gap-2">
          <button className="btn" onClick={onClose}>取消</button>
          <button className="btn btn-primary" onClick={submit} disabled={loading || !remoteServerId || !remoteRunDir}>
            {loading ? '处理中...' : '确认登记'}
          </button>
        </div>
      </div>
    </div>
  )
}
