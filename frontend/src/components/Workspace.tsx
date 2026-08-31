import { useCallback, useEffect, useRef, useState } from 'react'
import { Activity, Check, Edit2, FolderInput, RadioTower, Square, Trash2, X, Settings2, ZoomIn, ZoomOut } from 'lucide-react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'
import { getCurveColumns, getCurveMetricLabel, getDefaultCurveMetrics, getMetricSuffix } from '../curveMetrics'
import { ConfirmDialog } from './ConfirmDialog'
import { DeleteDialog } from './DeleteDialog'
import { ExperimentCurvesDialog } from './ExperimentCurvesDialog'
import { LocalTrialDialog } from './LocalTrialDialog'
import { ParameterEditor } from './ParameterEditor'
import { RemoteTrialDialog } from './RemoteTrialDialog'
import { TrialComparisonTable } from './TrialComparisonTable'
import { TrialSummaryDrawer } from './TrialSummaryDrawer'
import { TaskSettingsDialog } from './TaskSettingsDialog'

interface Props {
  experimentId: string
  onExperimentUpdated?: () => void
  onDeleted: () => void
}

const CANCELLABLE_STATUSES = new Set([
  'TRAINING',
])

const CHART_COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']

const formatDateTime = (value: string | undefined) => {
  if (!value) return '-'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

type YAxisMode = 'overview' | 'plateau'

const getYAxisDomain = (values: number[], mode: YAxisMode): [number, number] | [number, 'auto'] => {
  if (values.length === 0) return [0, 'auto']
  const min = Math.min(...values)
  const max = Math.max(...values)

  if (mode === 'overview') {
    const pad = Math.max(max * 0.05, 0.01)
    return [0, max + pad]
  }

  const range = max - min
  const pad = range > 0 ? range * 0.08 : Math.max(Math.abs(max) * 0.05, 0.01)
  return [Math.max(0, min - pad), max + pad]
}

const formatAxisTick = (value: number | string) => {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return String(value)
  if (Math.abs(numeric) >= 10) return numeric.toFixed(1).replace(/\.0$/, '')
  return numeric.toFixed(4).replace(/\.?0+$/, '')
}

const getPlatformValues = (rows: any[], trialId: string | undefined, metricKey: string) => {
  if (!trialId) return []
  const values = rows
    .map((point) => point[`${trialId}.${metricKey}`])
    .filter((value): value is number => typeof value === 'number')
  return values.slice(-Math.max(5, Math.ceil(values.length * 0.3)))
}

const shouldOfferForceDelete = (message: string) =>
  message.includes('--force') || message.includes('force=true')

export function Workspace({ experimentId, onExperimentUpdated, onDeleted }: Props) {
  const [detail, setDetail] = useState<any>(null)
  const [comparison, setComparison] = useState<any>(null)
  const [selectedTrialId, setSelectedTrialId] = useState<string | null>(null)
  const [trialToDelete, setTrialToDelete] = useState<{ trial_id: string; display_name?: string } | null>(null)
  const [loading, setLoading] = useState(true)
  const [isDeleting, setIsDeleting] = useState(false)
  const [isCancelling, setIsCancelling] = useState(false)
  const [showParameterDrawer, setShowParameterDrawer] = useState(false)
  const [chartData, setChartData] = useState<any[]>([])
  const [curveColumns, setCurveColumns] = useState<string[]>([])
  const [trialIds, setTrialIds] = useState<string[]>([])
  const [trialLabels, setTrialLabels] = useState<Record<string, string>>({})
  const [curveFitnessMetric, setCurveFitnessMetric] = useState('')
  const [showCurves, setShowCurves] = useState(false)
  const [showLocalDialog, setShowLocalDialog] = useState(false)
  const [showRemoteDialog, setShowRemoteDialog] = useState(false)
  const [showTaskSettings, setShowTaskSettings] = useState(false)
  const [isRenaming, setIsRenaming] = useState(false)
  const [renameValue, setRenameValue] = useState('')
  const [renaming, setRenaming] = useState(false)
  const [summaryMetricA, setSummaryMetricA] = useState('')
  const [summaryMetricB, setSummaryMetricB] = useState('')
  const [summaryYAxisMode, setSummaryYAxisMode] = useState<YAxisMode>('overview')
  const [hiddenSummaryTrials, setHiddenSummaryTrials] = useState<Set<string>>(new Set())

  const experimentIdRef = useRef(experimentId)
  experimentIdRef.current = experimentId

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const [det, comp, curvesData] = await Promise.all([
        api.getExperiment(experimentIdRef.current),
        api.getComparison(experimentIdRef.current),
        api.getExperimentCurves(experimentIdRef.current).catch(() => null),
      ])
      setDetail(det)
      setComparison(comp)
      
      if (curvesData?.curves) {
        const topTrials = Object.keys(curvesData.curves).sort().reverse().slice(0, 5)
        const columns = getCurveColumns(curvesData.curves)
        const [defaultMap, defaultRecall] = getDefaultCurveMetrics(columns, det.experiment.task_type)
        setTrialIds(topTrials)
        setTrialLabels(curvesData.trial_labels || {})
        setCurveFitnessMetric(curvesData.fitness_metric || '')
        setCurveColumns(columns)
        setSummaryMetricA((current) => columns.includes(current) ? current : defaultMap)
        setSummaryMetricB((current) => columns.includes(current) ? current : defaultRecall)
        const epochs = new Set<number>()
        Object.values(curvesData.curves).forEach((rows: any) => rows.forEach((row: any) => epochs.add(row.epoch)))
        const maxEpoch = Math.max(0, ...Array.from(epochs))
        
        const points = []
        for (let epoch = 1; epoch <= maxEpoch; epoch += 1) {
          const point: any = { epoch }
          topTrials.forEach((trialId) => {
            const row = curvesData.curves[trialId].find((item: any) => item.epoch === epoch)
            if (row) {
              columns.forEach((column) => {
                if (typeof row[column] === 'number') point[`${trialId}.${column}`] = row[column]
              })
            }
          })
          points.push(point)
        }
        setChartData(points)
      } else {
        setChartData([])
        setCurveColumns([])
        setTrialIds([])
        setTrialLabels({})
        setCurveFitnessMetric('')
      }
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    experimentIdRef.current = experimentId
    setHiddenSummaryTrials(new Set())
    setSummaryYAxisMode('overview')
    loadData()
  }, [experimentId, loadData])

  useEffect(() => {
    const remoteTrials = (detail?.trials || []).filter((trial: any) => trial.remote_server_id && ['TRAINING', 'RETRAINING'].includes(trial.internal_status))
    if (remoteTrials.length === 0) return undefined
    const timer = window.setInterval(() => {
      Promise.all(remoteTrials.map((trial: any) => api.syncRemoteTrial(trial.trial_id).catch(() => null))).then(() => loadData())
    }, 30000)
    return () => window.clearInterval(timer)
  }, [detail?.trials, loadData])

  const handleDeleteTrial = async (trialId: string, keepFiles: boolean) => {
    try {
      await api.deleteTrial(trialId, keepFiles, false)
      await loadData()
    } catch (err: any) {
      const msg = err?.detail?.error || '删除失败'
      if (shouldOfferForceDelete(msg) && confirm(`${msg}\n是否强制删除该训练记录？`)) {
        await api.deleteTrial(trialId, keepFiles, true)
        await loadData()
      } else {
        alert(msg)
      }
    }
  }

  const handleCancelExperiment = async () => {
    try {
      await api.cancelExperiment(experimentId, '用户手动停止')
      setIsCancelling(false)
      await loadData()
      onExperimentUpdated?.()
    } catch (err: any) {
      alert(err?.detail?.error || '停止失败')
    }
  }

  const handleRename = async () => {
    const nextName = renameValue.trim()
    if (!nextName) {
      alert('任务名称不能为空')
      return
    }
    setRenaming(true)
    try {
      await api.updateExperiment(experimentId, { description: nextName })
      setIsRenaming(false)
      await loadData()
      onExperimentUpdated?.()
    } catch (err: any) {
      alert(err?.detail?.error || '重命名失败')
    } finally {
      setRenaming(false)
    }
  }

  if (loading || !detail) return <div className="p-4">正在加载实验详情...</div>

  const experiment = detail.experiment
  const canCancel = CANCELLABLE_STATUSES.has(experiment.status)
  const [defaultMapMetric] = getDefaultCurveMetrics(curveColumns, experiment.task_type)
  const metricSuffix = getMetricSuffix(defaultMapMetric, experiment.task_type)
  const visibleSummaryTrialIds = trialIds.filter((trialId) => !hiddenSummaryTrials.has(trialId))

  const toggleSummaryTrial = (trialId: string) => {
    setHiddenSummaryTrials((current) => {
      const next = new Set(current)
      if (next.has(trialId)) next.delete(trialId)
      else next.add(trialId)
      return next
    })
  }

  const renderSummaryLegend = () => (
    <div className="flex items-center justify-center gap-2" style={{ flexWrap: 'wrap', fontSize: 11, paddingTop: 4 }}>
      {trialIds.map((trialId, index) => {
        const hidden = hiddenSummaryTrials.has(trialId)
        return (
          <button
            key={trialId}
            type="button"
            onClick={() => toggleSummaryTrial(trialId)}
            title={hidden ? '显示该 Trial 曲线' : '隐藏该 Trial 曲线'}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
              border: 'none',
              background: 'transparent',
              padding: '2px 4px',
              cursor: 'pointer',
              color: hidden ? 'var(--text-muted)' : CHART_COLORS[index % CHART_COLORS.length],
              textDecoration: hidden ? 'line-through' : 'none',
              opacity: hidden ? 0.65 : 1,
            }}
          >
            <span style={{ width: 8, height: 8, borderRadius: 999, background: CHART_COLORS[index % CHART_COLORS.length], opacity: hidden ? 0.35 : 1 }} />
            {trialLabels[trialId] || trialId}
          </button>
        )
      })}
    </div>
  )

  const renderSummaryChart = (metricKey: string, setMetricKey: (key: string) => void) => {
    const chartWindowData = chartData
    const visibleValues = chartData.flatMap((point) =>
      visibleSummaryTrialIds
        .map((trialId) => point[`${trialId}.${metricKey}`])
        .filter((value): value is number => typeof value === 'number')
    )
    const latestVisibleTrialId = visibleSummaryTrialIds[0] || trialIds[0]
    const platformValues = getPlatformValues(chartData, latestVisibleTrialId, metricKey)
    const yAxisValues = summaryYAxisMode === 'plateau' && platformValues.length > 0
      ? platformValues
      : visibleValues
    const yAxisDomain = getYAxisDomain(yAxisValues, summaryYAxisMode)

    return (
    <div className="inline-chart-card flex-1" style={{ height: 286, margin: 0, paddingBottom: 0 }}>
      <div className="flex items-center justify-between gap-2" style={{ marginBottom: '0.5rem', flexWrap: 'wrap' }}>
        <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>
          {getCurveMetricLabel(metricKey, curveFitnessMetric)} (最近 5 次 Trial)
        </div>
        <div className="flex items-center gap-2">
          <select className="input" style={{ width: 190, height: 34, fontSize: '0.8rem' }} value={metricKey} onChange={(event) => setMetricKey(event.target.value)}>
            {curveColumns.map((column) => (
              <option key={column} value={column}>{getCurveMetricLabel(column, curveFitnessMetric)}</option>
            ))}
          </select>
          <button
            className="btn"
            style={{ minWidth: 74, height: 34, padding: '0 0.55rem', fontSize: '0.78rem', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}
            onClick={() => setSummaryYAxisMode((mode) => mode === 'overview' ? 'plateau' : 'overview')}
            title={summaryYAxisMode === 'overview' ? '按最新训练任务的平台期放大纵轴' : '切换回总览范围'}
          >
            {summaryYAxisMode === 'overview' ? <><ZoomIn size={14} /> 平台期</> : <><ZoomOut size={14} /> 总览</>}
          </button>
        </div>
      </div>
      <ResponsiveContainer width="100%" height="88%">
        <LineChart data={chartWindowData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.2)" vertical={false} />
          <XAxis dataKey="epoch" tick={{fontSize: 11, fill: 'var(--text-muted)'}} axisLine={false} tickLine={false} />
          <YAxis domain={yAxisDomain} tickFormatter={formatAxisTick} tick={{fontSize: 11, fill: 'var(--text-muted)'}} axisLine={false} tickLine={false} width={52} allowDataOverflow={summaryYAxisMode === 'plateau'} />
          <Tooltip contentStyle={{ borderRadius: 8, border: 'none', boxShadow: 'var(--shadow-md)', background: 'rgba(255,255,255,0.95)' }} />
          <Legend content={renderSummaryLegend} />
          {visibleSummaryTrialIds.map((trialId) => {
            const index = trialIds.indexOf(trialId)
            return (
            <Line key={trialId} type="monotone" dataKey={`${trialId}.${metricKey}`} name={trialLabels[trialId] || trialId} stroke={CHART_COLORS[index % CHART_COLORS.length]} strokeWidth={2} dot={false} connectNulls activeDot={{ r: 4 }} />
            )
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
    )
  }

  return (
    <div className="workspace-grid">
      <div className="workspace-center">
        <div className="card">
          <div className="workspace-summary-header">
            <div style={{ minWidth: 0 }}>
              {isRenaming ? (
                <div className="flex items-center gap-2" style={{ marginBottom: '0.25rem' }}>
                  <input
                    className="input"
                    style={{ width: 320, maxWidth: '100%', fontSize: '1rem', fontWeight: 600 }}
                    value={renameValue}
                    onChange={(event) => setRenameValue(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') handleRename()
                      if (event.key === 'Escape') setIsRenaming(false)
                    }}
                    autoFocus
                  />
                  <button className="btn btn-primary" style={{ padding: '0.35rem 0.55rem' }} onClick={handleRename} disabled={renaming} title="保存">
                    <Check size={16} />
                  </button>
                  <button className="btn" style={{ padding: '0.35rem 0.55rem' }} onClick={() => setIsRenaming(false)} disabled={renaming} title="取消">
                    <X size={16} />
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-2" style={{ marginBottom: '0.25rem', flexWrap: 'wrap' }}>
                  <h1 style={{ fontSize: '1.25rem' }}>{experiment.description}</h1>
                  <button className="btn" style={{ padding: '0.25rem 0.45rem' }} onClick={() => setShowTaskSettings(true)} title="任务设置">
                    <Edit2 size={14} /> 任务设置
                  </button>
                </div>
              )}
              <div className="flex gap-4 text-muted" style={{ fontSize: '0.875rem', flexWrap: 'wrap' }}>
                <span>数据集：{experiment.dataset_root}</span>
                <span>默认模型：{detail.default_model || experiment.pretrained_model}</span>
                <span>最新训练：{formatDateTime(detail.latest_trial_created_at)}</span>
              </div>
            </div>
            <div className="workspace-summary-actions">
              <button className="btn btn-primary workspace-action-btn" onClick={() => setShowParameterDrawer(true)} title="训练参数设置">
                <Settings2 size={16} /> 参数设置
              </button>
              {canCancel && (
                <button className="btn workspace-action-btn" onClick={() => setIsCancelling(true)} title="停止任务">
                  <Square size={16} /> 停止任务
                </button>
              )}
              <button className="btn btn-danger workspace-action-btn" onClick={() => setIsDeleting(true)} title="删除任务">
                <Trash2 size={16} /> 删除任务
              </button>
            </div>
          </div>
        </div>

        <div className="card comparison-card">
          <div className="p-4" style={{ borderBottom: '1px solid var(--panel-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2 style={{ fontSize: '1rem' }}>历次试验对比汇总</h2>
            <div className="flex gap-2" style={{ flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              <button className="btn" onClick={() => setShowLocalDialog(true)}>
                <FolderInput size={16} /> 导入本地训练
              </button>
              <button className="btn" onClick={() => setShowRemoteDialog(true)}>
                <RadioTower size={16} /> 导入远程训练
              </button>
              <button className="btn" onClick={() => setShowCurves(true)}>
                <Activity size={16} /> 曲线对比
              </button>
              <button className="btn" onClick={loadData}>刷新</button>
            </div>
          </div>
          {chartData.length > 0 && (
            <div className="flex gap-4" style={{ margin: '1rem', marginBottom: 0 }}>
              {renderSummaryChart(summaryMetricA, setSummaryMetricA)}
              {renderSummaryChart(summaryMetricB, setSummaryMetricB)}
            </div>
          )}
          <div style={{ flex: 1, overflow: 'hidden', marginTop: chartData.length > 0 ? '1rem' : '0' }}>
            {comparison ? (
              <TrialComparisonTable
                data={comparison}
                metricSuffix={metricSuffix}
                onRowClick={setSelectedTrialId}
                onRequestDeleteTrial={setTrialToDelete}
              />
            ) : (
              <div className="p-4">暂无对比数据。</div>
            )}
          </div>
        </div>
      </div>

      {showParameterDrawer && (
        <>
          <div className="drawer-overlay" onClick={() => setShowParameterDrawer(false)} />
          <div className="drawer-content">
            <ParameterEditor
              experimentId={experimentId}
              onRunSuccess={async () => {
                await loadData()
                onExperimentUpdated?.()
              }}
              onClose={() => setShowParameterDrawer(false)}
            />
          </div>
        </>
      )}

      {selectedTrialId && (
        <TrialSummaryDrawer
          trialId={selectedTrialId}
          onClose={() => setSelectedTrialId(null)}
          onUpdated={loadData}
        />
      )}

      {showLocalDialog && (
        <LocalTrialDialog experimentId={experimentId} onClose={() => setShowLocalDialog(false)} onImported={loadData} />
      )}

      {showRemoteDialog && (
        <RemoteTrialDialog experimentId={experimentId} onClose={() => setShowRemoteDialog(false)} onImported={loadData} />
      )}
      {showTaskSettings && <TaskSettingsDialog experimentId={experimentId} onClose={() => setShowTaskSettings(false)} onSaved={async () => { await loadData(); onExperimentUpdated?.() }} />}

      {trialToDelete && (
        <DeleteDialog
          title="删除训练记录"
          message={`确定删除训练记录 ${trialToDelete.display_name || trialToDelete.trial_id} 吗？`}
          dangerousMessage="同时删除本地托管文件"
          onClose={() => setTrialToDelete(null)}
          onConfirm={async (keepFiles) => {
            await handleDeleteTrial(trialToDelete.trial_id, keepFiles)
            setTrialToDelete(null)
          }}
        />
      )}

      {isCancelling && (
        <ConfirmDialog
          title="停止任务"
          message="这会立即终止当前本地训练进程，并将实验状态标记为已取消。若当前没有本地训练进程在运行，则只会更新状态。"
          confirmLabel="确认停止"
          confirmClassName="btn btn-danger"
          onClose={() => setIsCancelling(false)}
          onConfirm={handleCancelExperiment}
        />
      )}

      {isDeleting && (
        <DeleteDialog
          title="删除任务"
          message={`确定删除实验"${experiment.description}"吗？此操作会移除该实验下的所有 Trial 记录。`}
          dangerousMessage="同时删除本地托管的训练文件"
          onClose={() => setIsDeleting(false)}
          onConfirm={async (keepFiles) => {
            try {
              await api.deleteExperiment(experimentId, keepFiles, false)
              setIsDeleting(false)
              onDeleted()
            } catch (err: any) {
              const msg = err?.detail?.error || '删除失败'
              if (shouldOfferForceDelete(msg) && confirm(`${msg}\n是否强制删除该实验？`)) {
                await api.deleteExperiment(experimentId, keepFiles, true)
                setIsDeleting(false)
                onDeleted()
              } else {
                alert(msg)
              }
            }
          }}
        />
      )}

      {showCurves && <ExperimentCurvesDialog experimentId={experimentId} taskType={experiment.task_type} onClose={() => setShowCurves(false)} />}
    </div>
  )
}
