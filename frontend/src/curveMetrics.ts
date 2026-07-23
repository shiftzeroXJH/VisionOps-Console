export type TaskType = 'detection' | 'segment' | 'obb' | string

type CurveRow = Record<string, unknown>
type Curves = Record<string, CurveRow[]>

const EXCLUDED_COLUMNS = new Set(['epoch', 'time'])

const metricColumn = (metric: string, suffix: string) =>
  suffix ? `metrics/${metric}(${suffix})` : `metrics/${metric}`

export const getCurveColumns = (curves: Curves | undefined): string[] => {
  const columns: string[] = []
  const seen = new Set<string>()

  Object.values(curves || {}).forEach((rows) => {
    rows.forEach((row) => {
      Object.entries(row).forEach(([key, value]) => {
        if (!EXCLUDED_COLUMNS.has(key) && typeof value === 'number' && Number.isFinite(value) && !seen.has(key)) {
          seen.add(key)
          columns.push(key)
        }
      })
    })
  })

  return columns
}

const taskSuffixes = (taskType: TaskType): string[] => {
  if (taskType === 'segment') return ['M']
  if (taskType === 'obb') return ['B', 'O', '']
  return ['B', '']
}

const findMetricColumn = (columns: string[], taskType: TaskType, metric: string) => {
  for (const suffix of taskSuffixes(taskType)) {
    const key = metricColumn(metric, suffix)
    if (columns.includes(key)) return key
  }
  return columns.find((column) => column.startsWith(`metrics/${metric}`))
}

export const getDefaultCurveMetrics = (columns: string[], taskType: TaskType): [string, string] => {
  const map = findMetricColumn(columns, taskType, 'mAP50-95') || columns[0] || ''
  const recall = findMetricColumn(columns, taskType, 'recall') || columns.find((column) => column !== map) || map
  return [map, recall]
}

export const getMetricSuffix = (metricColumnName: string, taskType: TaskType): string => {
  const match = metricColumnName.match(/\(([A-Za-z]+)\)$/)
  if (match) return `(${match[1]})`
  return taskSuffixes(taskType).find((suffix) => suffix) ? `(${taskSuffixes(taskType)[0]})` : ''
}
