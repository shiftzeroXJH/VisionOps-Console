import { Trash2 } from 'lucide-react'

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
    if (row.remote_training_status === 'maybe_stopped') {
      return (
        <span className="status-pill status-pill-danger" title="远程训练可能已停止">
          <span className="status-dot" />
          异常中断
        </span>
      )
    }
    switch (row.status) {
      case 'COMPLETED':
        return (
          <span className="status-pill status-pill-completed">
            <span className="status-dot" />
            完成
          </span>
        )
      case 'INTERRUPTED_OR_FAILED':
        return (
          <span className="status-pill status-pill-danger">
            <span className="status-dot" />
            失败
          </span>
        )
      case 'TRAINING':
        return (
          <span className="status-pill status-pill-running">
            <span className="status-dot" />
            训练中
          </span>
        )
      case 'QUEUED':
        return (
          <span className="status-pill status-pill-queued">
            <span className="status-dot" />
            排队中
          </span>
        )
      default:
        return <span className="status-pill status-pill-neutral"><span className="status-dot" />{row.status}</span>
    }
  }

  const renderDelta = (val: any) => {
    if (typeof val !== 'number') return <span className="font-mono text-muted">-</span>
    if (val > 0) return <span className="font-mono text-success" style={{ fontWeight: 600 }}>+{val.toFixed(4)}</span>
    if (val < 0) return <span className="font-mono text-danger" style={{ fontWeight: 600 }}>{val.toFixed(4)}</span>
    return <span className="font-mono text-muted">0.0000</span>
  }

  const isMonoColumn = (key: string) =>
    ['iteration', 'map50_95', 'fitness', 'map50', 'delta_map50_95', 'precision', 'recall',
     'best_epoch', 'epochs_completed', 'cumulative_epochs', 'imgsz', 'batch', 'lr0', 'patience'].includes(key)

  const cellValue = (row: any, key: string) => {
    if (key === 'status') return renderStatus(row)
    if (key === 'delta_map50_95') return renderDelta(row.delta_map50_95)
    if (key === 'display_name') {
      return (
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', fontWeight: 600 }}>
          {row.is_best && (
            <span className="status-pill status-pill-success" style={{ fontSize: '10px', padding: '0 4px', height: '16px', lineHeight: '16px' }}>
              BEST
            </span>
          )}
          <span>{row.display_name}</span>
        </span>
      )
    }
    if (key === 'training_mode') {
      return row.training_mode === 'continued' ? (
        <span style={{ display: 'inline-block', padding: '1px 5px', borderRadius: 3, background: '#f0fdfa', color: '#0f766e', border: '1px solid #ccfbf1', fontSize: '11px', fontWeight: 600 }}>
          续训
        </span>
      ) : (
        <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>常规</span>
      )
    }
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
            <th style={{ width: 44, textAlign: 'center' }}>操作</th>
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row: any) => (
            <tr
              key={row.trial_id}
              onClick={() => onRowClick(row.trial_id)}
              style={{
                cursor: 'pointer',
                backgroundColor: row.is_best ? 'rgba(5, 150, 105, 0.04)' : undefined,
              }}
            >
              {cols.map((col) => {
                const isMax = isMetricMaximum(row, col)
                const isMono = isMonoColumn(col)
                return (
                  <td
                    key={col}
                    className={isMono ? 'font-mono' : undefined}
                    style={{
                      fontWeight: isMax ? 700 : undefined,
                      color: isMax ? 'var(--primary)' : undefined,
                    }}
                  >
                    {cellValue(row, col)}
                  </td>
                )
              })}
              <td style={{ maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis' }}>{row.note || '-'}</td>
              <td style={{ textAlign: 'center' }}>
                <button
                  className="icon-btn"
                  style={{
                    width: 26,
                    height: 26,
                    padding: 0,
                    borderRadius: 4,
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: 'transparent',
                    color: 'var(--text-muted)',
                    border: 'none',
                    cursor: 'pointer',
                  }}
                  onClick={(event) => { event.stopPropagation(); onRequestDeleteTrial({ trial_id: row.trial_id, display_name: row.display_name }) }}
                  title="删除训练记录"
                  onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--danger-color)'; e.currentTarget.style.backgroundColor = '#fef2f2' }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.backgroundColor = 'transparent' }}
                >
                  <Trash2 size={14} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
