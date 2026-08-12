import { CheckCircle2, CircleDashed, Loader2, Trash2, XCircle } from 'lucide-react'

interface Props {
  data: any
  metricSuffix: string
  onRowClick: (trialId: string) => void
  onRequestDeleteTrial: (trial: { trial_id: string; display_name?: string }) => void
}

const HIGHLIGHT_METRICS = ['fitness', 'map50_95', 'map50', 'delta_map50_95', 'precision', 'recall'] as const

export function TrialComparisonTable({ data, metricSuffix, onRowClick, onRequestDeleteTrial }: Props) {
  if (!data?.rows?.length) {
    return <div className="p-4 text-muted">暂无训练记录，请先运行或导入一个 Trial。</div>
  }

  const cols = [
    'iteration', 'display_name', 'status',
    'map50_95', 'fitness', 'map50', 'delta_map50_95', 'precision', 'recall',
    'best_epoch', 'epochs_completed', 'cumulative_epochs', 'training_mode', 'model_display', 'imgsz', 'batch', 'lr0', 'patience',
  ]

  const metricMaximums = Object.fromEntries(
    HIGHLIGHT_METRICS.map((key) => [
      key,
      Math.max(
        ...data.rows
          .map((row: any) => row[key])
          .filter((value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)),
      ),
    ]),
  ) as Record<(typeof HIGHLIGHT_METRICS)[number], number>

  const isMetricMaximum = (row: any, key: string) =>
    HIGHLIGHT_METRICS.includes(key as (typeof HIGHLIGHT_METRICS)[number])
    && typeof row[key] === 'number'
    && Number.isFinite(row[key])
    && row[key] === metricMaximums[key as (typeof HIGHLIGHT_METRICS)[number]]

  const columnLabel = (key: string) => {
    const labels: Record<string, string> = {
      map50_95: `mAP50-95${metricSuffix}`,
      fitness: 'Fitness',
      map50: `mAP50${metricSuffix}`,
      delta_map50_95: `Delta mAP50-95${metricSuffix}`,
      precision: `Precision${metricSuffix}`,
      recall: `Recall${metricSuffix}`,
      cumulative_epochs: '累计 epochs',
      training_mode: '训练方式',
    }
    return labels[key] || (key === 'display_name' ? 'Trial' : key.replace(/_/g, ' '))
  }

  const formatValue = (val: any) => {
    if (val === undefined || val === null || val === '') return '-'
    if (typeof val === 'number') return Number.isInteger(val) ? val : val.toFixed(4)
    return val
  }

  const renderStatus = (row: any) => {
    const status = row.status
    if (row.remote_training_status === 'maybe_stopped') {
      return <span title="远程训练可能已停止"><XCircle size={16} className="text-danger" /></span>
    }
    switch (status) {
      case 'COMPLETED':
        return <CheckCircle2 size={16} className="text-success" />
      case 'INTERRUPTED_OR_FAILED':
        return <XCircle size={16} className="text-danger" />
      case 'TRAINING':
        return <Loader2 size={16} className="text-warning" style={{ animation: 'spin 1s linear infinite' }} />
      case 'QUEUED':
        return <CircleDashed size={16} className="text-warning" />
      default:
        return status
    }
  }

  const renderDelta = (val: any) => {
    if (typeof val !== 'number') return '-'
    if (val > 0) return <span className="text-success">+{val.toFixed(4)}</span>
    if (val < 0) return <span className="text-danger">{val.toFixed(4)}</span>
    return <span className="text-muted">0.0000</span>
  }

  const cellValue = (row: any, key: string) => {
    if (key === 'status') return renderStatus(row)
    if (key === 'delta_map50_95') return renderDelta(row.delta_map50_95)
    if (key === 'training_mode') return row.training_mode === 'continued' ? '续训' : '常规训练'
    if (['imgsz', 'batch', 'lr0', 'patience'].includes(key)) return formatValue(row.params?.[key])
    return formatValue(row[key])
  }

  return (
    <div className="table-wrapper h-full" style={{ border: 'none', borderRadius: 0 }}>
      <table>
        <thead style={{ position: 'sticky', top: 0, zIndex: 10 }}>
          <tr>
            {cols.map((col) => (
              <th key={col} title={col === 'fitness' ? data.fitness_metric : undefined}>
                {columnLabel(col)}
              </th>
            ))}
            <th>备注</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row: any) => (
            <tr
              key={row.trial_id}
              onClick={() => onRowClick(row.trial_id)}
              style={{ cursor: 'pointer', backgroundColor: row.is_best ? 'rgba(16,185,129,0.06)' : undefined }}
            >
              {cols.map((col) => (
                <td key={col} style={{ fontWeight: isMetricMaximum(row, col) ? 700 : undefined }}>
                  {cellValue(row, col)}
                </td>
              ))}
              <td style={{ maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis' }}>{row.note || '-'}</td>
              <td>
                <button
                  className="btn btn-danger"
                  style={{ padding: '0.2rem 0.4rem', backgroundColor: 'transparent', color: 'var(--danger-color)', border: 'none', boxShadow: 'none' }}
                  onClick={(event) => { event.stopPropagation(); onRequestDeleteTrial({ trial_id: row.trial_id, display_name: row.display_name }) }}
                  title="删除训练记录"
                >
                  <Trash2 size={16} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
