import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type TrainingTaskList, type RemoteGpuStatus } from '../api'

type Props = {
  experimentId: string
  params: Record<string, unknown>
  pretrained: string
  note: string
  onClose: () => void
  onStarted: () => void
}

export function RemoteTrainingDialog({ experimentId, params, pretrained, note, onClose, onStarted }: Props) {
  const [queues, setQueues] = useState<TrainingTaskList | null>(null)
  const [queueError, setQueueError] = useState('')
  const submissionRef = useRef<{ fingerprint: string; key: string } | null>(null)
  const submittingRef = useRef(false)
  const refreshQueues = useCallback(async () => {
    try { setQueues(await api.getTrainingTasks()); setQueueError('') }
    catch { setQueueError('无法获取队列数量，提交时将由服务器决定启动或排队。') }
  }, [])
  useEffect(() => { void refreshQueues() }, [refreshQueues])
  const [detail, setDetail] = useState<any>(null)
  const [servers, setServers] = useState<any[]>([])
  const [serverId, setServerId] = useState('')
  const selectedQueue = queues?.groups?.find((group) => group.source === 'remote' && group.remote_server_id === serverId)
  const willQueue = !!selectedQueue && (selectedQueue.running_count >= selectedQueue.max_parallel_training_tasks || selectedQueue.queued_count > 0 || selectedQueue.blocked)
  const [datasetRoot, setDatasetRoot] = useState('')
  const [datasetYaml, setDatasetYaml] = useState('')
  const [remoteModel, setRemoteModel] = useState(pretrained.split(/[\\/]/).pop() || pretrained)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [gpuStates, setGpuStates] = useState<Record<string, { data?: RemoteGpuStatus; error?: string; loading: boolean }>>({})
  const [refreshing, setRefreshing] = useState(false)
  const monitorRef = useRef<AbortController | null>(null)

  const stopMonitoring = useCallback(() => {
    monitorRef.current?.abort()
    monitorRef.current = null
  }, [])

  const refreshGpu = useCallback(async (items: any[]) => {
    if (monitorRef.current) return
    const controller = new AbortController()
    monitorRef.current = controller
    setRefreshing(true)
    setGpuStates((previous) => {
      const next = { ...previous }
      items.forEach((s) => { next[s.remote_server_id] = { ...previous[s.remote_server_id], loading: true } })
      return next
    })
    let index = 0
    const worker = async () => {
      while (!controller.signal.aborted && index < items.length) {
        const id = items[index++].remote_server_id
        try {
          const result = await api.getRemoteGpuStatus(id, controller.signal)
          if (controller.signal.aborted) return
          setGpuStates((previous) => ({ ...previous, [id]: {
            data: result.status === 'ok' ? result : previous[id]?.data,
            error: result.status === 'ok' ? undefined : result.error || '无法获取显存', loading: false,
          } }))
        } catch {
          if (controller.signal.aborted) return
          setGpuStates((previous) => ({ ...previous, [id]: { ...previous[id], error: '无法获取显存，请稍后重试', loading: false } }))
        }
      }
    }
    try {
      await Promise.all(Array.from({ length: Math.min(3, items.length) }, worker))
    } finally {
      if (monitorRef.current === controller) {
        monitorRef.current = null
        setRefreshing(false)
      }
    }
  }, [])

  useEffect(() => {
    if (servers.length) void refreshGpu(servers)
    return stopMonitoring
  }, [servers, refreshGpu, stopMonitoring])

  const close = () => { if (submittingRef.current) return; stopMonitoring(); onClose() }

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      const [experiment, serverData] = await Promise.all([api.getExperiment(experimentId), api.getRemoteServers()])
      if (cancelled) return
      setDetail(experiment)
      const items = serverData.remote_servers || []
      setServers(items)
      setServerId((current) => current || items[0]?.remote_server_id || '')
    }
    void load().catch((err) => { if (!cancelled) setError(err?.detail?.error || '加载远程配置失败') })
    return () => { cancelled = true }
  }, [experimentId])

  useEffect(() => {
    if (!serverId || !detail?.experiment) return
    const cfg = detail.experiment.remote_configs?.[serverId] || {}
    setDatasetRoot(cfg.dataset_root || '')
    setDatasetYaml(cfg.dataset_yaml || '')
    setRemoteModel(cfg.pretrained_model || (pretrained.split(/[\\/]/).pop() || pretrained))
  }, [serverId, detail, pretrained])

  const submit = async () => {
    if (!serverId || submittingRef.current) return
    const fingerprint = JSON.stringify({ experimentId, serverId, params, remoteModel: remoteModel.trim(), note, datasetRoot: datasetRoot.trim(), datasetYaml: datasetYaml.trim() })
    if (submissionRef.current?.fingerprint !== fingerprint) {
      submissionRef.current = { fingerprint, key: globalThis.crypto?.randomUUID?.() ?? Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, '0')).join('') }
    }
    submittingRef.current = true
    stopMonitoring()
    setRefreshing(false)
    setGpuStates((previous) => Object.fromEntries(Object.entries(previous).map(([id, state]) => [id, { ...state, loading: false }])))
    setBusy(true); setError('')
    try {
      const configs = { ...(detail?.experiment?.remote_configs || {}), [serverId]: { dataset_root: datasetRoot.trim(), dataset_yaml: datasetYaml.trim(), pretrained_model: remoteModel.trim() } }
      await api.updateExperiment(experimentId, { remote_configs: configs })
      const result = await api.runRemoteTrial(experimentId, { remote_server_id: serverId, params, pretrained: remoteModel.trim(), note, idempotency_key: submissionRef.current.key })
      window.dispatchEvent(new Event('training-tasks-changed'))
      const task = result.training_task
      const terminalStatus = task?.status === 'FAILED' ? '已失败'
        : task?.status === 'CANCELLED' ? '已取消'
        : task?.status === 'COMPLETED' ? '已完成' : null
      if (terminalStatus) {
        alert(`此前提交的训练任务${terminalStatus}，本次重试返回已有记录。${task.error ? `\n${task.error}` : ''}\n如需新建训练，请重新打开远程训练对话框并提交。`)
      } else {
        alert(result.disposition === 'queued'
          ? `训练已加入队列${task?.position ? `，当前第 ${task.position} 位` : ''}。`
          : '训练已提交，正在准备远程训练。')
      }
      onStarted(); onClose()
    } catch (err: any) { setError(err?.detail?.error || '远程训练提交失败，可重试') } finally { submittingRef.current = false; setBusy(false); void refreshQueues() }
  }

  return <div style={{ position: 'fixed', inset: 0, zIndex: 120, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.45)' }} onClick={close} />
    <div className="card flex-col gap-3" style={{ position: 'relative', width: 760, maxWidth: '96vw', maxHeight: '92vh', overflow: 'auto' }}>
      <div className="flex justify-between items-center"><h2 style={{ fontSize: '1.2rem' }}>远程训练</h2><button className="btn" onClick={close} disabled={busy}>关闭</button></div>
      {error && <div className="text-danger p-2">{error}</div>}
      <div className="flex justify-between items-center"><span id="remote-server-label">远程服务器</span><button className="btn" disabled={busy || refreshing || !servers.length} onClick={() => { void refreshGpu(servers); void refreshQueues() }}>{refreshing ? '查询显存中…' : '刷新显存'}</button></div>
      {queueError && <div role="status" className="text-warning">{queueError}</div>}
      <div role="radiogroup" aria-labelledby="remote-server-label" className="flex-col gap-2" style={{ maxHeight: 300, overflowY: 'auto' }}>
        {!servers.length && <div className="text-muted">暂无服务器，请先在设置中配置。</div>}
        {servers.map((s) => {
          const queue = queues?.groups?.find((group) => group.source === 'remote' && group.remote_server_id === s.remote_server_id)
          const state = gpuStates[s.remote_server_id]
          const data = state?.data
          const percent = data?.memory_used_percent ?? 0
          return <label key={s.remote_server_id} className="card" style={{ padding: '0.75rem', cursor: 'pointer', borderColor: serverId === s.remote_server_id ? 'var(--primary, #3b82f6)' : undefined }}>
            <div className="flex items-center gap-2"><input type="radio" name="remote-training-server" value={s.remote_server_id} checked={serverId === s.remote_server_id} disabled={busy} onChange={() => setServerId(s.remote_server_id)} /><strong>{s.name}</strong><span className="text-muted" style={{ fontSize: 12, overflowWrap: 'anywhere' }}>{s.username}@{s.host}:{s.port}</span></div>
            <div className="text-muted" style={{ fontSize: 13, marginTop: 6 }}>运行 {queue?.running_count ?? '—'} / {queue?.max_parallel_training_tasks ?? s.max_parallel_training_tasks ?? 1} · 排队 {queue?.queued_count ?? '—'}{queue?.blocked ? ' · 队列阻塞' : ''}</div>
            {data && <>
              <div style={{ marginTop: 6, fontSize: 13 }}>{data.gpu_name} · 显存占用 {percent.toFixed(1)}% · 剩余 {((data.memory_free_mib || 0) / 1024).toFixed(1)} / {((data.memory_total_mib || 0) / 1024).toFixed(1)} GiB</div>
              <div role="progressbar" aria-label={`${s.name} 显存占用`} aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100} style={{ height: 6, marginTop: 6, background: 'var(--border-color, #e5e7eb)', borderRadius: 4 }}><div style={{ width: `${percent}%`, height: '100%', borderRadius: 4, background: percent >= 90 ? '#ef4444' : percent >= 70 ? '#f59e0b' : '#3b82f6' }} /></div>
              <div className="text-muted" style={{ fontSize: 12, marginTop: 4 }}>采集时间：{data.captured_at ? new Date(data.captured_at).toLocaleTimeString('zh-CN', { hour12: false }) : '—'}{state?.error ? '（旧数据）' : ''}</div>
            </>}
            {state?.loading && <div className="text-muted" style={{ fontSize: 12 }}>正在查询显存…</div>}
            {state?.error && <div className="text-warning" style={{ fontSize: 12 }}>无法获取：{state.error}</div>}
          </label>
        })}
      </div>
      <div className="text-muted" style={{ fontSize: 12 }}>显存为查询时的参考值；打开时查询一次，可手动刷新。查询失败仍可选择服务器启动训练。</div>
      <label>远程数据集目录<input className="input" disabled={busy} value={datasetRoot} onChange={(e) => setDatasetRoot(e.target.value)} placeholder="留空则上传本地数据集" /></label>
      <label>远程数据集 YAML（目录和 YAML 均留空时自动上传本地数据集）<input className="input" disabled={busy} value={datasetYaml} onChange={(e) => setDatasetYaml(e.target.value)} placeholder=".../data.yaml" /></label>
      <label>远程默认模型<input className="input" disabled={busy} value={remoteModel} onChange={(e) => setRemoteModel(e.target.value)} /></label>
      <div className="text-muted" style={{ fontSize: 12 }}>数据集目录和 YAML 均留空时，平台会把实验的本地数据集上传到远程工作目录并自动改写 YAML。AMP 检查权重会自动上传到每个 Trial 目录。</div>
      <div className="flex justify-end gap-2"><button className="btn" onClick={close} disabled={busy}>取消</button><button className="btn btn-primary" onClick={() => void submit()} disabled={busy || !serverId || !remoteModel.trim()}>{busy ? '提交中...' : willQueue ? '加入队列' : '提交训练'}</button></div>
    </div>
  </div>
}
