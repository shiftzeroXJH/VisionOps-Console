import { useCallback, useEffect, useRef, useState } from 'react'
import { Download, Edit2, RefreshCw, RotateCcw, Save, ScanSearch, X } from 'lucide-react'
import { api } from '../api'
import { ExportOnnxDialog } from './ExportOnnxDialog'
import { ImageGallery } from './ImageGallery'
import { ContinueTrainingDialog } from './ContinueTrainingDialog'
import { SaveHyperparameterTemplateDialog } from './SaveHyperparameterTemplateDialog'

interface Props {
  trialId: string
  onClose: () => void
  onUpdated?: () => void
}

const formatDateTime = (value: string | undefined) => {
  if (!value) return '-'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

const PARAM_DISPLAY_ORDER = [
  'imgsz', 'batch', 'epochs', 'patience', 'workers',
  'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'cos_lr',
  'degrees', 'translate', 'scale', 'shear', 'perspective', 'flipud', 'fliplr',
  'hsv_h', 'hsv_s', 'hsv_v',
]
const LOW_FREQUENCY_PARAMS = ['erasing', 'copy_paste', 'mixup', 'mosaic']

const sortParamEntries = (params: Record<string, unknown>) => {
  const commonPriority = new Map(PARAM_DISPLAY_ORDER.map((key, index) => [key, index]))
  const lowFrequencyPriority = new Map(LOW_FREQUENCY_PARAMS.map((key, index) => [key, index]))
  const getRank = (key: string) => {
    const commonRank = commonPriority.get(key)
    if (commonRank !== undefined) return commonRank
    const lowFrequencyRank = lowFrequencyPriority.get(key)
    return lowFrequencyRank === undefined
      ? PARAM_DISPLAY_ORDER.length
      : PARAM_DISPLAY_ORDER.length + 1 + lowFrequencyRank
  }

  return Object.entries(params).sort(([left], [right]) => {
    const leftRank = getRank(left)
    const rightRank = getRank(right)
    if (leftRank !== rightRank) return leftRank - rightRank
    return left.localeCompare(right)
  })
}

export function TrialSummaryDrawer({ trialId, onClose, onUpdated }: Props) {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [syncing, setSyncing] = useState(false)
  const [showExportDialog, setShowExportDialog] = useState(false)
  const [showContinueDialog, setShowContinueDialog] = useState(false)
  const [editingName, setEditingName] = useState(false)
  const [draftName, setDraftName] = useState('')
  const [savingName, setSavingName] = useState(false)
  const [showSaveTemplateDialog, setShowSaveTemplateDialog] = useState(false)
  const [templateNotice, setTemplateNotice] = useState('')

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
  const displayName = trial.display_name || trial.trial_id || trialId
  const continuation = data?.continuation || {}

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

  const openModelEvaluation = () => {
    const params = new URLSearchParams({
      trial_id: trialId,
      dataset_path: String(trial.dataset_yaml || datasetAnalysis.dataset_yaml || ''),
      imgsz: String(trial.imgsz || data?.params?.imgsz || 640),
    })
    window.location.hash = `#/workbench/evaluation?${params.toString()}`
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
            {trial.training_mode === 'continued' && (
              <span className="continuation-badge">续训自 {trial.parent_display_name || trial.parent_trial_id}</span>
            )}
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
          <div className="flex gap-2" style={{ flexWrap: 'wrap', justifyContent: 'flex-end' }}>
            <button
              className="btn btn-primary"
              onClick={() => setShowContinueDialog(true)}
              disabled={loading || !continuation.can_continue}
              title={continuation.can_continue ? '从 last.pt 追加训练' : continuation.unavailable_reason || '当前不可续训'}
            >
              <RotateCcw size={16} /> 继续训练
            </button>
            <button className="btn" onClick={openModelEvaluation} disabled={loading || !data}>
              <ScanSearch size={16} /> 模型评估
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
                  {data.per_class_metrics?.length > 0 ? (
                    <div className="table-wrapper">
                      <table>
                        <thead>
                          <tr>
                            <th>class id</th>
                            <th>class name</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>mAP50</th>
                            <th>mAP50-95</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[...data.per_class_metrics]
                            .sort((left: any, right: any) => left.class_id - right.class_id)
                            .map((item: any) => (
                              <tr key={item.class_id}>
                                <td>{item.class_id}</td>
                                <td>{item.class_name}</td>
                                {['precision', 'recall', 'map50', 'map50_95'].map((key) => (
                                  <td key={key}>{typeof item[key] === 'number' ? item[key].toFixed(4) : '-'}</td>
                                ))}
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="text-muted" style={{ fontSize: '0.8rem' }}>暂无类别指标</div>
                  )}
                </section>

                <section className="flex-col gap-2">
                  <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>资源占用 / 来源同步</h3>
                  <div className="table-wrapper">
                    <table>
                      <tbody>
                        <tr><td className="text-muted">模型</td><td>{trial.model_display || trial.model || '-'}</td></tr>
                        <tr><td className="text-muted">模型来源</td><td>{trial.model_source || '-'}</td></tr>
                        <tr><td className="text-muted">参数来源</td><td>{trial.params_source || '-'}</td></tr>
                        <tr><td className="text-muted">任务状态</td><td>{trial.status || '-'}</td></tr>
                        {trial.remote_training_status && <tr><td className="text-muted">远程训练状态</td><td>{trial.remote_training_status}</td></tr>}
                        <tr><td className="text-muted">同步状态</td><td>{trial.sync_status || '-'}</td></tr>
                        <tr><td className="text-muted">最近同步</td><td>{trial.last_synced_at || '-'}</td></tr>
                        <tr><td className="text-muted">已同步 epoch</td><td>{trial.last_synced_epoch_count ?? '-'}</td></tr>
                        <tr><td className="text-muted">训练开始</td><td>{formatDateTime(trial.created_at)}</td></tr>
                        <tr><td className="text-muted">本地目录</td><td style={{ fontSize: '0.75rem', wordBreak: 'break-all' }}>{trial.run_dir || '-'}</td></tr>
                        <tr><td className="text-muted">远程目录</td><td style={{ fontSize: '0.75rem', wordBreak: 'break-all' }}>{trial.remote_run_dir || '-'}</td></tr>
                        {trial.sync_error && <tr><td className="text-muted">同步错误</td><td className="text-danger">{trial.sync_error}</td></tr>}
                      </tbody>
                    </table>
                  </div>
                </section>

                <section className="flex-col gap-2">
                  <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>数据集分析</h3>
                  {isRemote && !datasetAnalysis.totals && (
                    <div className="text-muted">此任务未保存训练时的数据集统计快照，刷新不会使用当前数据集补算历史统计。</div>
                  )}
                  {isRemote && datasetAnalysis.totals && (
                    <div className="text-muted">已保存的数据集统计快照，刷新远程数据不会重新计算。</div>
                  )}
                  {datasetAnalysis.totals && (
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
                  )}
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
                  <div className="trial-params-heading">
                    <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>参数</h3>
                    <div className="flex items-center gap-2">
                      {templateNotice && <span className="template-save-notice">{templateNotice}</span>}
                      <button className="btn" onClick={() => setShowSaveTemplateDialog(true)}>
                        <Save size={16} /> 保存为模板
                      </button>
                    </div>
                  </div>
                  <div className="trial-params-grid">
                    {sortParamEntries(data.params || {}).map(([key, value]) => (
                      <div className="trial-param-item" key={key}>
                        <span className="text-muted">{key}</span>
                        <span className="trial-param-value">{typeof value === 'number' ? value : String(value)}</span>
                      </div>
                    ))}
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
          defaultOutputDir={trial.default_export_dir}
          onClose={() => setShowExportDialog(false)}
        />
      )}
      {showContinueDialog && (
        <ContinueTrainingDialog
          trialId={trialId}
          onClose={() => setShowContinueDialog(false)}
          onSubmitted={async () => {
            await load()
            onUpdated?.()
          }}
        />
      )}
      {showSaveTemplateDialog && (
        <SaveHyperparameterTemplateDialog
          trialId={trialId}
          defaultName={displayName}
          onClose={() => setShowSaveTemplateDialog(false)}
          onSaved={(name, overwritten) => {
            setShowSaveTemplateDialog(false)
            setTemplateNotice(overwritten ? `已覆盖模板：${name}` : `已保存模板：${name}`)
          }}
        />
      )}
    </>
  )
}
