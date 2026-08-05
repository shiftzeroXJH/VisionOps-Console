import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, X } from 'lucide-react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'
import { getCurveColumns, getCurveMetricLabel, getDefaultCurveMetrics, type TaskType } from '../curveMetrics'

interface Props {
  experimentId: string
  taskType: TaskType
  onClose: () => void
}

const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']

type YAxisMode = 'overview' | 'tail'

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

const getChartWindowData = (rows: any[], trialIds: string[], metricKey: string, mode: YAxisMode) => {
  if (mode === 'overview') return rows
  const rowsWithVisibleValues = rows.filter((point) =>
    trialIds.some((trialId) => typeof point[trialId]?.[metricKey] === 'number')
  )
  return rowsWithVisibleValues.slice(-Math.max(5, Math.ceil(rowsWithVisibleValues.length * 0.3)))
}

export function ExperimentCurvesDialog({ experimentId, taskType, onClose }: Props) {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [selectedTrials, setSelectedTrials] = useState<Set<string>>(new Set())
  const [selectedMetricA, setSelectedMetricA] = useState('')
  const [selectedMetricB, setSelectedMetricB] = useState('')
  const [yAxisMode, setYAxisMode] = useState<YAxisMode>('overview')

  useEffect(() => {
    const fetchCurves = async () => {
      setLoading(true)
      try {
        const res = await api.getExperimentCurves(experimentId)
        setData(res)
        setSelectedTrials(new Set(Object.keys(res.curves || {}).sort().reverse().slice(0, 5)))
        const [defaultMap, defaultRecall] = getDefaultCurveMetrics(getCurveColumns(res.curves), taskType)
        setSelectedMetricA(defaultMap)
        setSelectedMetricB(defaultRecall)
      } finally {
        setLoading(false)
      }
    }
    fetchCurves().catch(console.error)
  }, [experimentId, taskType])

  const toggleTrial = (trialId: string) => {
    const next = new Set(selectedTrials)
    if (next.has(trialId)) next.delete(trialId)
    else next.add(trialId)
    setSelectedTrials(next)
  }

  const trialIds = useMemo(() => Object.keys(data?.curves || {}).sort(), [data])
  const trialLabels = useMemo(() => data?.trial_labels || {}, [data])
  const curveColumns = useMemo(() => getCurveColumns(data?.curves), [data])

  const colorMap = useMemo(() => {
    const map = new Map<string, string>()
    trialIds.forEach((id, index) => map.set(id, COLORS[index % COLORS.length]))
    return map
  }, [trialIds])

  const chartData = useMemo(() => {
    if (!data?.curves) return []
    const epochs = new Set<number>()
    Object.values(data.curves).forEach((rows: any) => rows.forEach((row: any) => epochs.add(row.epoch)))
    const maxEpoch = Math.max(0, ...Array.from(epochs))
    const points = []
    for (let epoch = 1; epoch <= maxEpoch; epoch += 1) {
      const point: any = { epoch }
      Object.keys(data.curves).forEach((trialId) => {
        const row = data.curves[trialId].find((item: any) => item.epoch === epoch)
        if (row) point[trialId] = row
      })
      points.push(point)
    }
    return points
  }, [data])

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (event.key === 'Escape') onClose()
  }, [onClose])

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  const renderMetricChart = (chartId: string, metricKey: string, setMetricKey: (key: string) => void) => {
    const selectedTrialIds = trialIds.filter((trialId) => selectedTrials.has(trialId))
    const chartWindowData = getChartWindowData(chartData, selectedTrialIds, metricKey, yAxisMode)
    const yAxisValues = chartWindowData.flatMap((point: any) =>
      selectedTrialIds
        .map((trialId) => point[trialId]?.[metricKey])
        .filter((value): value is number => typeof value === 'number')
    )
    const yAxisDomain = getYAxisDomain(yAxisValues, yAxisMode)

    return (
      <div key={chartId} style={{ minHeight: 420, border: '1px solid var(--panel-border)', borderRadius: 8, padding: '1rem' }}>
        <div className="flex items-center justify-between gap-2" style={{ marginBottom: '1rem', flexWrap: 'wrap' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, margin: 0 }}>{getCurveMetricLabel(metricKey, data?.fitness_metric)}</h3>
          <select className="input" style={{ width: 360, maxWidth: '100%', height: 34, fontSize: '0.8rem' }} value={metricKey} onChange={(event) => setMetricKey(event.target.value)}>
            {curveColumns.map((column) => <option key={column} value={column}>{getCurveMetricLabel(column, data?.fitness_metric)}</option>)}
          </select>
        </div>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={chartWindowData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.25)" />
            <XAxis dataKey="epoch" />
            <YAxis domain={yAxisDomain} tickFormatter={formatAxisTick} width={56} allowDataOverflow={yAxisMode === 'tail'} />
            <Tooltip />
            <Legend />
            {selectedTrialIds.map((trialId) => (
              <Line key={trialId} type="monotone" dataKey={`${trialId}.${metricKey}`} name={trialLabels[trialId] || trialId} stroke={colorMap.get(trialId)} strokeWidth={2} dot={false} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    )
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 60, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ position: 'absolute', inset: 0, backgroundColor: 'rgba(0,0,0,0.5)' }} onClick={onClose} />
      <div className="card flex-col" style={{ position: 'relative', width: '90vw', height: '85vh', maxWidth: 1400, zIndex: 61, padding: 0, display: 'flex', overflow: 'hidden' }}>
        <div className="flex justify-between items-center p-4" style={{ borderBottom: '1px solid var(--panel-border)' }}>
          <h2 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity className="text-primary" /> 曲线对比
          </h2>
          <div className="flex items-center gap-2">
            <button
              className="btn"
              onClick={() => setYAxisMode((mode) => mode === 'overview' ? 'tail' : 'overview')}
              title={yAxisMode === 'overview' ? '切换到尾段范围' : '切换到总览范围'}
            >
              {yAxisMode === 'overview' ? '尾段' : '总览'}
            </button>
            <button className="btn" style={{ padding: '0.25rem' }} onClick={onClose}><X size={20} /></button>
          </div>
        </div>

        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          <div style={{ width: 240, borderRight: '1px solid var(--panel-border)', padding: '1rem', overflowY: 'auto' }}>
            <h3 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '1rem' }} className="text-muted">选择 Trial</h3>
            {trialIds.map((trialId) => (
              <label key={trialId} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.875rem', marginBottom: '0.5rem' }}>
                <input type="checkbox" checked={selectedTrials.has(trialId)} onChange={() => toggleTrial(trialId)} />
                <span style={{ color: selectedTrials.has(trialId) ? colorMap.get(trialId) : 'var(--text-muted)' }}>{trialLabels[trialId] || trialId}</span>
              </label>
            ))}
          </div>

          <div style={{ flex: 1, padding: '1.5rem', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            {loading ? (
              <div className="text-center text-muted">加载中...</div>
            ) : chartData.length === 0 || curveColumns.length === 0 ? (
              <div className="text-center text-muted">暂无可绘制曲线。</div>
            ) : (
              <>
                {renderMetricChart('primary', selectedMetricA, setSelectedMetricA)}
                {renderMetricChart('secondary', selectedMetricB, setSelectedMetricB)}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
