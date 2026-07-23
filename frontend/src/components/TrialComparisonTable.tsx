import { CheckCircle2, CircleDashed, Loader2, Trash2, XCircle } from 'lucide-react'

interface Props {
  data: any
  metricSuffix: string
  onRowClick: (trialId: string) => void
  onRequestDeleteTrial: (trial: { trial_id: string; display_name?: string }) => void
}

export function TrialComparisonTable({ data, metricSuffix, onRowClick, onRequestDeleteTrial }: Props) {
  const sourceLabel = (source: string) => {
    switch (source) {
      case 'trained':
        return '本地训练'
      case 'imported':
        return '本地导入'
      case 'remote_sftp':
        return '远程导入'
      default:
        return source || '-'
    }
  }

  if (!data?.rows?.length) {
    return <div className="p-4 text-muted">暂无训练记录，请先运行或导入一个 Trial。</div>
  }

  const cols = [
    'iteration', 'display_name', 'status', 'model_display', 'source', 'server',
    'map50_95', 'map50', 'delta_map50_95', 'precision', 'recall',
    'best_epoch', 'epochs_completed', 'imgsz', 'batch', 'lr0', 'patience',
  ]

  const columnLabel = (key: string) => {
    const labels: Record<string, string> = {
      map50_95: `mAP50-95${metricSuffix}`,
      map50: `mAP50${metricSuffix}`,
      delta_map50_95: `Delta mAP50-95${metricSuffix}`,
      precision: `Precision${metricSuffix}`,
      recall: `Recall${metricSuffix}`,
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
      return <span title="远程训练可能已停止"><CircleDashed size={16} className="text-warning" /></span>
    }
    switch (status) {
      case 'COMPLETED':
        return <CheckCircle2 size={16} className="text-success" />
      case 'FAILED':
      case 'CANCELLED':
        return <XCircle size={16} className="text-danger" />
      case 'TRAINING':
      case 'RETRAINING':
      case 'ANALYZING':
        return <Loader2 size={16} className="text-warning" style={{ animation: 'spin 1s linear infinite' }} />
      case 'WAITING_USER_CONFIRM':
      case 'AUTO_TUNE_PENDING':
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
    if (['imgsz', 'batch', 'lr0', 'patience'].includes(key)) return formatValue(row.params?.[key])
    if (key === 'source') return <span className="badge">{sourceLabel(row.source)}</span>
    return formatValue(row[key])
  }

  return (
    <div className="table-wrapper h-full" style={{ border: 'none', borderRadius: 0 }}>
      <table>
        <thead style={{ position: 'sticky', top: 0, zIndex: 10 }}>
          <tr>
            {cols.map((col) => <th key={col}>{columnLabel(col)}</th>)}
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
                <td key={col} style={{ fontWeight: col === 'map50_95' || (col === 'display_name' && row.is_best) ? 700 : undefined }}>
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
