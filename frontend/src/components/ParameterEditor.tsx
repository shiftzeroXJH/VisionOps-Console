import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronDown, ChevronUp, Plus, Settings2, Trash2 } from 'lucide-react'
import { api } from '../api'

interface Props {
  experimentId: string
  onRunSuccess: () => void
  onClose?: () => void
}

type ParamGroup = {
  id: string
  title: string
  description: string
  keys: string[]
}

type ExtraParam = {
  id: string
  key: string
  value: string
}

type ApiError = {
  detail?: {
    code?: string
    running_count?: number
    max_parallel_training_tasks?: number
  }
}

const PARAM_GROUPS: ParamGroup[] = [
  {
    id: 'core',
    title: '训练规模',
    description: '控制输入尺寸、批大小、训练轮数和早停策略。',
    keys: ['imgsz', 'batch', 'workers', 'epochs', 'patience'],
  },
  {
    id: 'optimizer',
    title: '优化器',
    description: '学习率、优化器类型以及预热和衰减相关配置。',
    keys: ['optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'cos_lr'],
  },
  {
    id: 'geometry',
    title: '几何增强',
    description: '适合 AOI 场景的轻量几何扰动，避免破坏目标结构。',
    keys: ['degrees', 'translate', 'scale', 'shear', 'perspective', 'flipud', 'fliplr'],
  },
  {
    id: 'appearance',
    title: '颜色与拼接增强',
    description: '颜色扰动和拼接增强，默认保持偏保守。',
    keys: ['hsv_h', 'hsv_s', 'hsv_v', 'mosaic', 'mixup', 'copy_paste', 'erasing'],
  },
]

const PARAM_LABELS: Record<string, string> = {
  imgsz: 'imgsz',
  batch: 'batch',
  workers: 'workers',
  epochs: 'epochs',
  patience: 'patience',
  optimizer: 'optimizer',
  lr0: 'lr0',
  lrf: 'lrf',
  momentum: 'momentum',
  weight_decay: 'weight_decay',
  warmup_epochs: 'warmup_epochs',
  cos_lr: 'cos_lr',
  degrees: 'degrees（旋转角度）',
  translate: 'translate（平移比例）',
  scale: 'scale（缩放幅度）',
  shear: 'shear（错切幅度）',
  perspective: 'perspective（透视变换）',
  flipud: 'flipud（上下翻转）',
  fliplr: 'fliplr（左右翻转）',
  hsv_h: 'hsv_h（色相扰动）',
  hsv_s: 'hsv_s（饱和度扰动）',
  hsv_v: 'hsv_v（明度扰动）',
  mosaic: 'mosaic（马赛克增强）',
  mixup: 'mixup（样本混合）',
  copy_paste: 'copy_paste（复制粘贴增强）',
  erasing: 'erasing（随机擦除）',
}

const DEFAULT_EXPANDED: Record<string, boolean> = {
  core: true,
  optimizer: true,
  geometry: false,
  appearance: false,
}

const displayExtraValue = (value: any) => typeof value === 'string' ? value : JSON.stringify(value)

const parseExtraValue = (value: string, schemaType?: string): { value?: any; error?: string } => {
  if (schemaType === 'string') return { value }
  if (schemaType === 'boolean') {
    if (value === 'true') return { value: true }
    if (value === 'false') return { value: false }
    return { error: '布尔参数只能选择 true 或 false' }
  }
  if (schemaType === 'integer' || schemaType === 'number') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? { value: parsed } : { error: '请输入有效数值' }
  }
  try {
    return { value: JSON.parse(value) }
  } catch {
    return { error: '请输入有效 JSON 值' }
  }
}

export function ParameterEditor({ experimentId, onRunSuccess, onClose }: Props) {
  const [schemaData, setSchemaData] = useState<any>(null)
  const [params, setParams] = useState<Record<string, any>>({})
  const [extraParams, setExtraParams] = useState<ExtraParam[]>([])
  const [model, setModel] = useState('')
  const [note, setNote] = useState('')
  const [loading, setLoading] = useState(false)
  const [validating, setValidating] = useState(false)
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState<Record<string, boolean>>(DEFAULT_EXPANDED)

  const experimentIdRef = useRef(experimentId)
  experimentIdRef.current = experimentId

  const loadParams = useCallback(async () => {
    const data = await api.getParams(experimentIdRef.current)
    const fixedSchema = data.editable_schema || {}
    const latest = data.latest_params || data.initial_params || {}
    const fixed: Record<string, any> = {}
    const extras: ExtraParam[] = []
    Object.entries(latest).forEach(([key, value]) => {
      if (fixedSchema[key]) fixed[key] = value
      else extras.push({ id: `${key}-${crypto.randomUUID()}`, key, value: displayExtraValue(value) })
    })
    setSchemaData(data)
    setParams(fixed)
    setExtraParams(extras)
    setModel(data.default_model || '')
    setExpanded(DEFAULT_EXPANDED)
    setValidationErrors({})
  }, [])

  useEffect(() => {
    loadParams().catch(console.error)
  }, [experimentId, loadParams])

  const buildCandidateParams = (): { params?: Record<string, any>; errors: Record<string, string> } => {
    const candidate = { ...params }
    const errors: Record<string, string> = {}
    const extraSchema = schemaData?.extra_param_schema || {}
    const usedKeys = new Set<string>()
    extraParams.forEach((row) => {
      const key = row.key.trim()
      if (!key) {
        errors[`extra-${row.id}`] = '参数名不能为空'
        return
      }
      if (usedKeys.has(key)) {
        errors[`extra-${row.id}`] = '额外参数名重复'
        return
      }
      usedKeys.add(key)
      if (Object.hasOwn(candidate, key)) {
        errors[`extra-${row.id}`] = '该参数已有固定控件，请在上方修改'
        return
      }
      if (!extraSchema[key]) {
        errors[`extra-${row.id}`] = '请选择当前 YOLO 支持的参数名'
        return
      }
      const parsed = parseExtraValue(row.value, extraSchema[key].type)
      if (parsed.error) {
        errors[`extra-${row.id}`] = parsed.error
        return
      }
      candidate[key] = parsed.value
    })
    return Object.keys(errors).length ? { errors } : { params: candidate, errors }
  }

  const handleValidate = async () => {
    const candidate = buildCandidateParams()
    if (!candidate.params) {
      setValidationErrors(candidate.errors)
      return false
    }
    setValidating(true)
    setValidationErrors({})
    try {
      const res = await api.validateParams(experimentId, candidate.params)
      if (!res.valid) {
        setValidationErrors(res.errors || {})
        return false
      }
      return true
    } catch (err: any) {
      setValidationErrors({ general: err?.detail?.error || '参数校验失败' })
      return false
    } finally {
      setValidating(false)
    }
  }

  const handleRun = async () => {
    const isValid = await handleValidate()
    const candidate = buildCandidateParams()
    if (!isValid || !candidate.params) return
    setLoading(true)
    try {
      try {
        await api.runTrial(experimentId, {
          params: candidate.params,
          pretrained: model,
          note,
          reason: 'Manual tuning',
        })
      } catch (err: unknown) {
        const detail = (err as ApiError)?.detail
        if (detail?.code !== 'TRAINING_CAPACITY_REACHED') throw err
        const running = Number(detail.running_count) || 0
        const maximum = Number(detail.max_parallel_training_tasks) || 1
        if (!confirm(`当前已有 ${running}/${maximum} 个训练任务运行中，是否加入排队队列？`)) return
        await api.runTrial(experimentId, {
          params: candidate.params,
          pretrained: model,
          note,
          reason: 'Manual tuning',
          enqueue_if_busy: true,
        })
      }
      setNote('')
      await onRunSuccess()
      onClose?.()
    } catch (err: any) {
      alert(err?.detail?.error || err?.message || '运行失败')
    } finally {
      setLoading(false)
    }
  }

  const updateParam = (key: string, value: any) => {
    setParams((current) => ({ ...current, [key]: value }))
  }

  const updateExtra = (id: string, patch: Partial<ExtraParam>) => {
    setExtraParams((current) => current.map((row) => row.id === id ? { ...row, ...patch } : row))
  }

  const toggleGroup = (groupId: string) => {
    setExpanded((current) => ({ ...current, [groupId]: !current[groupId] }))
  }

  if (!schemaData) return <div className="card h-full">正在加载参数...</div>

  const schema = schemaData.editable_schema || {}
  const extraSchema = schemaData.extra_param_schema || {}
  const renderField = (key: string) => {
    const field = schema[key]
    if (!field) return null
    const isError = validationErrors[key]
    const label = PARAM_LABELS[key] || key
    const helper = field.type === 'int' || field.type === 'float'
      ? `[${field.min ?? ''} - ${field.max ?? ''}]`
      : field.type === 'choice' && field.values?.length
      ? field.values.join(' / ')
      : ''

    return (
      <div key={key} className="param-field">
        <label className="param-label">
          <span>{label}</span>
          <span className="param-helper">{helper}</span>
        </label>
        {field.type === 'choice' ? (
          <select
            className="input"
            style={{ borderColor: isError ? 'var(--danger-color)' : undefined }}
            value={params[key] ?? ''}
            onChange={(event) => {
              const raw = event.target.value
              const sample = field.values?.[0]
              updateParam(key, typeof sample === 'number' ? Number(raw) : typeof sample === 'boolean' ? raw === 'true' : raw)
            }}
          >
            <option value="" disabled>{`选择 ${label}`}</option>
            {field.values?.map((value: any) => <option key={String(value)} value={String(value)}>{String(value)}</option>)}
          </select>
        ) : (
          <input
            type="number"
            step={field.type === 'int' ? field.step || 1 : 'any'}
            className="input"
            style={{ borderColor: isError ? 'var(--danger-color)' : undefined }}
            value={params[key] ?? ''}
            onChange={(event) => updateParam(key, field.type === 'int' ? parseInt(event.target.value, 10) : parseFloat(event.target.value))}
          />
        )}
        {isError && <span className="text-danger" style={{ fontSize: '0.7rem' }}>{String(isError)}</span>}
      </div>
    )
  }

  return (
    <div className="h-full flex-col parameter-editor-shell" style={{ display: 'flex', overflow: 'hidden', padding: '1.5rem', background: 'transparent', boxShadow: 'none', border: 'none' }}>
      <div className="parameter-header">
        <div>
          <h2 className="parameter-title">本地训练参数</h2>
          <p className="parameter-subtitle">按训练规模、优化器和增强策略分组管理，适合 AOI 检测场景。</p>
        </div>
        <div className="parameter-badge">
          <Settings2 size={16} />
          <span>{Object.keys(schema).length + extraParams.length} 项</span>
        </div>
      </div>

      <div className="parameter-scroll">
        {validationErrors.general && <div className="text-danger p-2">{validationErrors.general}</div>}
        <div className="parameter-model-row">
          <label className="param-label"><span>模型路径</span><span className="param-helper">支持本地 `.pt` 权重或项目内模型名</span></label>
          <input className="input" value={model} onChange={(event) => setModel(event.target.value)} />
        </div>

        <div className="parameter-groups">
          {PARAM_GROUPS.map((group) => {
            const isOpen = expanded[group.id]
            return (
              <section key={group.id} className="param-group-card">
                <button type="button" className="param-group-toggle" onClick={() => toggleGroup(group.id)}>
                  <div><div className="param-group-title">{group.title}</div><div className="param-group-description">{group.description}</div></div>
                  {isOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                </button>
                {isOpen && <div className="param-grid">{group.keys.map((key) => renderField(key))}</div>}
              </section>
            )
          })}
        </div>

        <section className="param-group-card" style={{ marginTop: '0.75rem' }}>
          <div className="param-group-toggle" style={{ cursor: 'default' }}>
            <div><div className="param-group-title">额外 YOLO 参数</div><div className="param-group-description">从提示中选择参数名，系统会自动识别数据类型；仅应用于本次 Trial。</div></div>
            <button type="button" className="btn" title="添加额外参数" onClick={() => setExtraParams((current) => [...current, { id: crypto.randomUUID(), key: '', value: '' }])}><Plus size={16} /></button>
          </div>
          <datalist id="extra-yolo-parameters">{Object.keys(extraSchema).map((key) => <option key={key} value={key} />)}</datalist>
          {extraParams.map((row) => {
            const schemaType = extraSchema[row.key]?.type
            return (
              <div key={row.id} className="param-grid" style={{ gridTemplateColumns: 'minmax(140px, 1fr) minmax(160px, 1fr) auto', alignItems: 'start', padding: '0.75rem 1rem 0' }}>
                <input className="input" list="extra-yolo-parameters" placeholder="参数名，例如 close_mosaic" value={row.key} onChange={(event) => updateExtra(row.id, { key: event.target.value })} />
                {schemaType === 'boolean' ? (
                  <select className="input" value={row.value} onChange={(event) => updateExtra(row.id, { value: event.target.value })}><option value="">选择值</option><option value="true">true</option><option value="false">false</option></select>
                ) : (
                  <input className="input" placeholder={schemaType === 'json' ? 'JSON 值，例如 [0, 1]' : '参数值'} value={row.value} onChange={(event) => updateExtra(row.id, { value: event.target.value })} />
                )}
                <button type="button" className="btn" title="删除额外参数" onClick={() => setExtraParams((current) => current.filter((item) => item.id !== row.id))}><Trash2 size={16} /></button>
                {validationErrors[`extra-${row.id}`] && <span className="text-danger" style={{ gridColumn: '1 / -1', fontSize: '0.7rem' }}>{validationErrors[`extra-${row.id}`]}</span>}
                {validationErrors[row.key] && <span className="text-danger" style={{ gridColumn: '1 / -1', fontSize: '0.7rem' }}>{validationErrors[row.key]}</span>}
              </div>
            )
          })}
        </section>
      </div>

      <div className="parameter-footer">
        <div className="flex-col gap-2 mb-4">
          <label className="param-label"><span>备注</span><span className="param-helper">可选，用于记录这次调参的目的</span></label>
          <input className="input" value={note} onChange={(event) => setNote(event.target.value)} />
        </div>
        <div className="flex gap-2">
          {onClose && <button className="btn flex-1" onClick={onClose} disabled={validating || loading}>关闭</button>}
          <button className="btn btn-primary flex-1" onClick={handleRun} disabled={loading || validating}>{loading ? '正在启动...' : '开始训练（自动校验）'}</button>
        </div>
      </div>
    </div>
  )
}
