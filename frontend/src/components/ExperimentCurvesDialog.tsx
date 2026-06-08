import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, X } from 'lucide-react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'

interface Props {
  experimentId: string
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

export function ExperimentCurvesDialog({ experimentId, onClose }: Props) {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [selectedTrials, setSelectedTrials] = useState<Set<string>>(new Set())
  const [yAxisMode, setYAxisMode] = useState<YAxisMode>('overview')

  useEffect(() => {
    const fetchCurves = async () => {
      setLoading(true)
      try {
        const res = await api.getExperimentCurves(experimentId)
        setData(res)
        setSelectedTrials(new Set(Object.keys(res.curves || {}).sort().reverse().slice(0, 5)))
      } finally {
        setLoading(false)
      }
    }
    fetchCurves().catch(console.error)
  }, [experimentId])

  const toggleTrial = (trialId: string) => {
    const next = new Set(selectedTrials)
    if (next.has(trialId)) next.delete(trialId)
    else next.add(trialId)
    setSelectedTrials(next)
  }

  const trialIds = useMemo(() => Object.keys(data?.curves || {}).sort(), [data])
  const trialLabels = useMemo(() => data?.trial_labels || {}, [data])

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

  const metrics = [
    { key: 'metrics/mAP50-95(B)', label: 'Box mAP50-95' },
    { key: 'metrics/mAP50(B)', label: 'Box mAP50' },
    { key: 'metrics/recall(B)', label: 'Recall' },
    { key: 'metrics/precision(B)', label: 'Precision' },
    { key: 'train/box_loss', label: 'Train Box Loss' },
    { key: 'train/cls_loss', label: 'Train Class Loss' },
    { key: 'val/box_loss', label: 'Val Box Loss' },
  ]

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (event.key === 'Escape') onClose()
  }, [onClose])

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

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
            ) : chartData.length === 0 ? (
              <div className="text-center text-muted">暂无可绘制曲线。</div>
            ) : (
              metrics.map((metric) => {
                const hasMetric = chartData.some((point: any) =>
                  trialIds.some((trialId) => selectedTrials.has(trialId) && point[trialId] && typeof point[trialId][metric.key] === 'number')
                )
                if (!hasMetric) return null
                const selectedTrialIds = trialIds.filter((trialId) => selectedTrials.has(trialId))
                const chartWindowData = getChartWindowData(chartData, selectedTrialIds, metric.key, yAxisMode)
                const yAxisValues = chartWindowData.flatMap((point: any) =>
                  selectedTrialIds
                    .map((trialId) => point[trialId]?.[metric.key])
                    .filter((value): value is number => typeof value === 'number')
                )
                const yAxisDomain = getYAxisDomain(yAxisValues, yAxisMode)
                return (
                  <div key={metric.key} style={{ minHeight: 420, border: '1px solid var(--panel-border)', borderRadius: 8, padding: '1rem' }}>
                    <h3 style={{ fontSize: '1rem', fontWeight: 600, textAlign: 'center' }}>{metric.label}</h3>
                    <ResponsiveContainer width="100%" height="90%">
                      <LineChart data={chartWindowData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.25)" />
                        <XAxis dataKey="epoch" />
                        <YAxis domain={yAxisDomain} tickFormatter={formatAxisTick} width={56} allowDataOverflow={yAxisMode === 'tail'} />
                        <Tooltip />
                        <Legend />
                        {trialIds.filter((trialId) => selectedTrials.has(trialId)).map((trialId) => (
                          <Line key={trialId} type="monotone" dataKey={`${trialId}.${metric.key}`} name={trialLabels[trialId] || trialId} stroke={colorMap.get(trialId)} strokeWidth={2} dot={false} connectNulls />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )
              })
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
