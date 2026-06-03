import { useCallback, useEffect, useRef, useState } from 'react'
import { Download, Edit2, Eye, RefreshCw, X } from 'lucide-react'
import { api } from '../api'
import { ExportOnnxDialog } from './ExportOnnxDialog'
import { ImageGallery } from './ImageGallery'
import { ValidationPreviewDialog } from './ValidationPreviewDialog'

interface Props {
  trialId: string
  onClose: () => void
  onUpdated?: () => void
}

export function TrialSummaryDrawer({ trialId, onClose, onUpdated }: Props) {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [syncing, setSyncing] = useState(false)
  const [showExportDialog, setShowExportDialog] = useState(false)
  const [showValidationDialog, setShowValidationDialog] = useState(false)
  const [editingName, setEditingName] = useState(false)
  const [draftName, setDraftName] = useState('')
  const [savingName, setSavingName] = useState(false)

  const trialIdRef = useRef(trialId)
  trialIdRef.current = trialId

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const summary = await api.getTrialSummary(trialIdRef.current)
      setData(summary)
      setDraftName(summary?.trial?.display_name || summary?.trial?.trial_id || '')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load().catch(console.error)
  }, [trialId, load])

  const syncRemote = async () => {
    setSyncing(true)
    try {
      await api.syncRemoteTrial(trialId)
      await load()
      onUpdated?.()
    } catch (err: any) {
      alert(err?.detail?.error || '远程同步失败')
    } finally {
      setSyncing(false)
    }
  }

  const trial = data?.trial || {}
  const datasetAnalysis = data?.dataset_analysis || {}
  const datasetSplits = datasetAnalysis.splits || {}
  const datasetTotals = datasetAnalysis.totals || {}
  const datasetClasses = Array.isArray(datasetAnalysis.classes) ? datasetAnalysis.classes : []
  const datasetWarnings = Array.isArray(datasetAnalysis.warnings) ? datasetAnalysis.warnings : []
  const isRemote = trial.source === 'remote_sftp'
  const architecture = String(trial.model_display || trial.model || 'model').replace(/\.pt$/i, '')
  const displayName = trial.display_name || trial.trial_id || trialId

  const saveTrialName = async () => {
    const nextName = draftName.trim()
    if (!nextName) {
      alert('Trial 名称不能为空')
      return
    }
    setSavingName(true)
    try {
      await api.renameTrial(trialId, { display_name: nextName })
      setEditingName(false)
      await load()
      onUpdated?.()
    } catch (err: any) {
      alert(err?.detail?.error || '重命名 Trial 失败')
    } finally {
      setSavingName(false)
    }
  }

  return (
    <>
      <div style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.3)', zIndex: 40 }} onClick={onClose} />
      <div
        style={{
          position: 'fixed', right: 0, top: 0, bottom: 0, width: 1300, maxWidth: '95vw',
          zIndex: 50, backgroundColor: 'var(--panel-bg)', display: 'flex', flexDirection: 'column',
          boxShadow: '-4px 0 24px rgba(0,0,0,0.6)', borderLeft: '1px solid var(--panel-border)',
        }}
      >
        <div className="flex justify-between items-center p-4" style={{ borderBottom: '1px solid var(--panel-border)' }}>
          <div className="flex items-center gap-2" style={{ minWidth: 0 }}>
            <h2 style={{ fontSize: '1.25rem', margin: 0 }}>
              Trial <span className="text-primary">{displayName}</span>
            </h2>
            {editingName ? (
              <div className="flex items-center gap-2">
                <input
                  className="input"
                  style={{ width: 280 }}
                  value={draftName}
                  onChange={(event) => setDraftName(event.target.value)}
                  disabled={savingName}
                />
                <button className="btn btn-primary" onClick={saveTrialName} disabled={savingName}>
                  {savingName ? '保存中...' : '保存'}
                </button>
                <button
                  className="btn"
                  onClick={() => {
                    setEditingName(false)
                    setDraftName(displayName)
                  }}
                  disabled={savingName}
                >
                  取消
                </button>
              </div>
            ) : (
              <button className="btn" onClick={() => setEditingName(true)} disabled={loading}>
                <Edit2 size={16} /> 编辑
              </button>
            )}
          </div>
          <div className="flex gap-2">
            <button className="btn" onClick={() => setShowValidationDialog(true)} disabled={loading}>
              <Eye size={16} /> 验证
            </button>
            <button className="btn" onClick={() => setShowExportDialog(true)} disabled={loading}>
              <Download size={16} /> 导出 ONNX
            </button>
            {isRemote && (
              <button className="btn" onClick={syncRemote} disabled={syncing}>
                <RefreshCw size={16} /> {syncing ? '同步中...' : '刷新远程数据'}
              </button>
            )}
            <button className="btn" style={{ padding: '0.25rem' }} onClick={onClose}><X size={20} /></button>
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', minHeight: 0, paddingBottom: '2rem', overscrollBehavior: 'contain' }}>
          {loading ? (
            <div className="text-muted p-4">正在加载报告...</div>
          ) : !data ? (
            <div className="text-danger p-4">未能加载报告。</div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.4fr) minmax(0, 1fr)', gap: '2rem', padding: '1.5rem', alignItems: 'start' }}>
              <div className="flex-col gap-4" style={{ minWidth: 0 }}>
                <h3 style={{ fontSize: '1.25rem', fontWeight: 600 }}>可视化</h3>
                <ImageGallery trialId={trialId} />
              </div>

              <div className="flex-col gap-6">
                {data.warnings?.length > 0 && (
                  <div className="card" style={{ backgroundColor: 'rgba(245,158,11,0.06)' }}>
                    <h3 className="text-warning mb-2" style={{ fontSize: '1rem' }}>警告</h3>
                    <ul style={{ paddingLeft: '1.25rem', fontSize: '0.875rem' }} className="text-muted">
                      {data.warnings.map((warning: string, index: number) => <li key={index}>{warning}</li>)}
                    </ul>
                  </div>
                )}

                <section className="flex-col gap-2">
                  <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>当前指标</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem' }}>
                    {Object.entries(data.final_metrics || {}).map(([key, value]: [string, any]) => (
                      <div key={key} className="card p-2 text-center" style={{ padding: '0.75rem' }}>
                        <div className="text-muted" style={{ fontSize: '0.75rem', textTransform: 'uppercase' }}>{key}</div>
                        <div style={{ fontSize: '1.25rem', fontWeight: 600 }}>{typeof value === 'number' ? value.toFixed(4) : value}</div>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="flex-col gap-2">
                  <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>资源占用 / 来源同步</h3>
                  <div className="table-wrapper">
                    <table>
                      <tbody>
                        <tr><td className="text-muted">模型</td><td>{trial.model_display || trial.model || '-'}</td></tr>
                        <tr><td className="text-muted">模型来源</td><td>{trial.model_source || '-'}</td></tr>
                        <tr><td className="text-muted">参数来源</td><td>{trial.params_source || '-'}</td></tr>
                        <tr><td className="text-muted">训练状态</td><td>{trial.remote_training_status || trial.status || '-'}</td></tr>
                        <tr><td className="text-muted">同步状态</td><td>{trial.sync_status || '-'}</td></tr>
                        <tr><td className="text-muted">最近同步</td><td>{trial.last_synced_at || '-'}</td></tr>
                        <tr><td className="text-muted">已同步 epoch</td><td>{trial.last_synced_epoch_count ?? '-'}</td></tr>
                        <tr><td className="text-muted">显存峰值</td><td>{data.resource?.gpu_mem_peak ? `${data.resource.gpu_mem_peak} MB` : '-'}</td></tr>
                        <tr><td className="text-muted">本地目录</td><td style={{ fontSize: '0.75rem', wordBreak: 'break-all' }}>{trial.run_dir || '-'}</td></tr>
                        <tr><td className="text-muted">远程目录</td><td style={{ fontSize: '0.75rem', wordBreak: 'break-all' }}>{trial.remote_run_dir || '-'}</td></tr>
                        {trial.sync_error && <tr><td className="text-muted">同步错误</td><td className="text-danger">{trial.sync_error}</td></tr>}
                      </tbody>
                    </table>
                  </div>
                </section>

                <section className="flex-col gap-2">
                  <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>数据集分析</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem' }}>
                    <div className="card p-2 text-center" style={{ padding: '0.75rem' }}>
                      <div className="text-muted" style={{ fontSize: '0.75rem' }}>Train 实例数</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 600 }}>{datasetTotals.train_instances ?? 0}</div>
                      <div className="text-muted" style={{ fontSize: '0.75rem', marginTop: '0.2rem' }}>
                        图像 {datasetSplits.train?.image_count ?? 0} / 标签 {datasetSplits.train?.label_file_count ?? 0}
                      </div>
                    </div>
                    <div className="card p-2 text-center" style={{ padding: '0.75rem' }}>
                      <div className="text-muted" style={{ fontSize: '0.75rem' }}>Val 实例数</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 600 }}>{datasetTotals.val_instances ?? 0}</div>
                      <div className="text-muted" style={{ fontSize: '0.75rem', marginTop: '0.2rem' }}>
                        图像 {datasetSplits.val?.image_count ?? 0} / 标签 {datasetSplits.val?.label_file_count ?? 0}
                      </div>
                    </div>
                    <div className="card p-2 text-center" style={{ padding: '0.75rem' }}>
                      <div className="text-muted" style={{ fontSize: '0.75rem' }}>总计 / 类别数</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 600 }}>{datasetTotals.total_instances ?? 0}</div>
                      <div className="text-muted" style={{ fontSize: '0.75rem', marginTop: '0.2rem' }}>
                        类别 {datasetTotals.class_count ?? 0}
                      </div>
                    </div>
                  </div>
                  {datasetAnalysis.dataset_yaml && (
                    <div className="text-muted" style={{ fontSize: '0.75rem', wordBreak: 'break-all' }}>
                      数据集配置: {datasetAnalysis.dataset_yaml}
                    </div>
                  )}
                  {datasetWarnings.length > 0 && (
                    <div className="card" style={{ backgroundColor: 'rgba(245,158,11,0.06)' }}>
                      <h4 className="text-warning mb-2" style={{ fontSize: '0.95rem' }}>数据集告警</h4>
                      <ul style={{ paddingLeft: '1.25rem', fontSize: '0.875rem' }} className="text-muted">
                        {datasetWarnings.map((warning: string, index: number) => <li key={index}>{warning}</li>)}
                      </ul>
                    </div>
                  )}
                  {datasetClasses.length > 0 ? (
                    <div className="table-wrapper">
                      <table>
                        <thead>
                          <tr>
                            <th>class id</th>
                            <th>class name</th>
                            <th>train</th>
                            <th>val</th>
                            <th>total</th>
                            <th>ratio</th>
                          </tr>
                        </thead>
                        <tbody>
                          {datasetClasses.map((item: any) => (
                            <tr key={item.class_id}>
                              <td>{item.class_id}</td>
                              <td>{item.class_name}</td>
                              <td>{item.train_instances}</td>
                              <td>{item.val_instances}</td>
                              <td>{item.total_instances}</td>
                              <td>{typeof item.total_ratio === 'number' ? `${(item.total_ratio * 100).toFixed(2)}%` : '-'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="text-muted">当前 Trial 未保存可展示的数据集类别分布。</div>
                  )}
                </section>

                <section className="flex-col gap-2">
                  <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>参数</h3>
                  <div className="table-wrapper">
                    <table>
                      <tbody>
                        {Object.entries(data.params || {}).map(([key, value]: [string, any]) => (
                          <tr key={key}>
                            <td className="text-muted" style={{ width: '50%' }}>{key}</td>
                            <td>{typeof value === 'number' ? value : String(value)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              </div>
            </div>
          )}
        </div>
      </div>
      {showExportDialog && (
        <ExportOnnxDialog
          trialId={trialId}
          modelStem={displayName}
          imgsz={Number(trial.imgsz || data?.params?.imgsz || 0)}
          architecture={architecture}
          onClose={() => setShowExportDialog(false)}
        />
      )}

      {showValidationDialog && (
        <ValidationPreviewDialog
          trial={{ trial_id: trialId, display_name: displayName }}
          onClose={() => setShowValidationDialog(false)}
        />
      )}
    </>
  )
}
