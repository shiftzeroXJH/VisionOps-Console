import { useEffect, useState } from 'react'
import { ArrowLeft, Plus, Save, Settings2, Trash2, X } from 'lucide-react'
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
  const [remoteServers, setRemoteServers] = useState<any[]>([])
  const [serverForm, setServerForm] = useState({ name: '', host: '', port: '22', username: '', password: '', remote_python: '', default_runs_root: '' })
  const [serverBusy, setServerBusy] = useState(false)
  const [serverMessage, setServerMessage] = useState('')
  const [editingServer, setEditingServer] = useState<any | null>(null)

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
        const remote = await api.getRemoteServers()
        if (!cancelled) setRemoteServers(remote.remote_servers || [])
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

  const saveRemoteServer = async () => {
    setServerBusy(true); setError(''); setServerMessage('')
    try {
      const payload = { ...serverForm, port: Number(serverForm.port), auth_type: 'password' }
      const result = editingServer?.remote_server_id
        ? await api.updateRemoteServer(editingServer.remote_server_id, payload)
        : await api.createRemoteServer(payload)
      setRemoteServers((current) => editingServer?.remote_server_id
        ? current.map((item) => item.remote_server_id === editingServer.remote_server_id ? result.remote_server : item)
        : [result.remote_server, ...current])
      setServerForm({ name: '', host: '', port: '22', username: '', password: '', remote_python: '', default_runs_root: '' })
      setEditingServer(null)
      setServerMessage(editingServer?.remote_server_id ? '远程服务器设置已更新。' : '远程服务器已保存。')
    } catch (err: any) { setError(err?.detail?.error || '保存远程服务器失败') } finally { setServerBusy(false) }
  }

  const testRemoteServer = async (serverId: string) => {
    setServerBusy(true); setError(''); setServerMessage('')
    try { await api.testRemoteServer(serverId); setServerMessage('连接测试成功。') } catch (err: any) { setError(err?.detail?.error || '连接测试失败') } finally { setServerBusy(false) }
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

        <div className="settings-content-grid">
          <div className="settings-main-column">
            <section className="settings-section settings-section-stack">
              <label className="settings-form-block"><h3>最大并行模型训练任务</h3><p className="text-muted">仅限制平台启动的本地调参训练；降低数值不会停止正在运行的任务。</p><input className="input" type="number" min={1} max={64} step={1} value={maxParallelTrainingTasks} onChange={(event) => setMaxParallelTrainingTasks(Number(event.target.value))} disabled={loading || saving} /></label>
              <button className="btn btn-primary" onClick={saveSettings} disabled={loading || saving}><Save size={16} /> {saving ? '保存中...' : '保存设置'}</button>
            </section>
            <section className="settings-section"><div><h3>验证与工作台缓存</h3><p className="text-muted">清除验证预览、上传图片和临时推理结果；不会删除训练记录或预测 XML。</p></div><button className="btn btn-danger" onClick={clearValidationCache} disabled={clearing}><Trash2 size={16} /> {clearing ? '清除中...' : '清除缓存'}</button></section>
          </div>
          <section className="settings-section settings-section-stack settings-remote-column">
            {!editingServer ? <>
              <div><h3>远程服务器</h3><p className="text-muted">服务器凭据在这里统一维护，训练时只选择服务器。</p></div>
              <div className="flex-col gap-2">{remoteServers.map((server) => <div key={server.remote_server_id} className="settings-server-row"><div><strong>{server.name}</strong><span className="text-muted">{server.username}@{server.host}:{server.port}</span></div><div className="flex gap-2"><button className="icon-btn" title="测试连接" onClick={() => void testRemoteServer(server.remote_server_id)} disabled={serverBusy}>✓</button><button className="icon-btn" title="服务器设置" onClick={() => { setServerMessage(''); setEditingServer(server); setServerForm({ name: server.name, host: server.host, port: String(server.port), username: server.username, password: '', remote_python: server.remote_python || '', default_runs_root: server.default_runs_root || '' }) }}><Settings2 size={16} /></button></div></div>)}{remoteServers.length === 0 && <div className="text-muted">尚未配置服务器</div>}</div>
              <button className="btn" onClick={() => { setServerMessage(''); setEditingServer({}); setServerForm({ name: '', host: '', port: '22', username: '', password: '', remote_python: '', default_runs_root: '' }) }}><Plus size={16} /> 新增服务器</button>
            </> : <>
              <div className="flex items-center gap-2"><button className="icon-btn" title="返回服务器列表" onClick={() => setEditingServer(null)}><ArrowLeft size={16} /></button><div><h3>{editingServer.remote_server_id ? '服务器设置' : '新增服务器'}</h3><p className="text-muted">密码认证信息保存在本地配置中。</p></div></div>
              <div className="flex-col gap-2">{([['name','名称'],['host','Host'],['port','端口'],['username','用户名'],['password','SSH 密码（留空保持不变）'],['remote_python','远程 Python 路径'],['default_runs_root','远程工作目录']] as const).map(([key, placeholder]) => <input key={key} className="input" type={key === 'password' ? 'password' : 'text'} placeholder={placeholder} value={serverForm[key]} onChange={(event) => setServerForm({ ...serverForm, [key]: event.target.value })} />)}</div>
              <div className="flex gap-2"><button className="btn" onClick={() => setEditingServer(null)}>取消</button><button className="btn btn-primary" onClick={() => void saveRemoteServer()} disabled={serverBusy || !serverForm.host || !serverForm.username || (!editingServer.remote_server_id && !serverForm.password) || !serverForm.remote_python || !serverForm.default_runs_root}><Save size={16} /> 保存服务器</button></div>
            </>}
            {serverMessage && <div className="text-success">{serverMessage}</div>}
          </section>
        </div>

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
