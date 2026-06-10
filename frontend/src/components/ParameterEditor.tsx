import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronDown, ChevronUp, Settings2 } from 'lucide-react'
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

const waitForTrainingJob = async (jobId: string) => {
  const deadline = Date.now() + 30000
  while (Date.now() < deadline) {
    const job = await api.getJob(jobId)
    if (job.status === 'failed') throw new Error(job.error || '训练启动失败')
    if (job.status === 'completed') return job
    await new Promise((resolve) => window.setTimeout(resolve, 800))
  }
  return null
}

export function ParameterEditor({ experimentId, onRunSuccess, onClose }: Props) {
  const [schemaData, setSchemaData] = useState<any>(null)
  const [params, setParams] = useState<any>({})
  const [model, setModel] = useState('')
  const [note, setNote] = useState('')
  const [loading, setLoading] = useState(false)
  const [validating, setValidating] = useState(false)
  const [validationErrors, setValidationErrors] = useState<any>({})
  const [expanded, setExpanded] = useState<Record<string, boolean>>(DEFAULT_EXPANDED)

  const experimentIdRef = useRef(experimentId)
  experimentIdRef.current = experimentId

  const loadParams = useCallback(async () => {
    const data = await api.getParams(experimentIdRef.current)
    setSchemaData(data)
    setParams(data.latest_params || data.initial_params || {})
    setModel(data.default_model || '')
    setExpanded(DEFAULT_EXPANDED)
  }, [])

  useEffect(() => {
    loadParams().catch(console.error)
  }, [experimentId, loadParams])

  const handleValidate = async () => {
    setValidating(true)
    setValidationErrors({})
    try {
      const res = await api.validateParams(experimentId, params)
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
    if (!isValid) return
    setLoading(true)
    try {
      const job = await api.runTrial(experimentId, { params, pretrained: model, note, reason: 'Manual tuning' })
      if (job?.job_id) await waitForTrainingJob(job.job_id)
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
    setParams((current: any) => ({ ...current, [key]: value }))
  }

  const toggleGroup = (groupId: string) => {
    setExpanded((current) => ({ ...current, [groupId]: !current[groupId] }))
  }

  if (!schemaData) return <div className="card h-full">正在加载参数...</div>

  const schema = schemaData.editable_schema || {}
  const groupedKeys = new Set(PARAM_GROUPS.flatMap((group) => group.keys))
  const extraKeys = Object.keys(schema).filter((key) => !groupedKeys.has(key))
  const groups = extraKeys.length
    ? [...PARAM_GROUPS, { id: 'extra', title: '其他参数', description: '当前实验额外暴露的可调参数。', keys: extraKeys }]
    : PARAM_GROUPS

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
              let parsed: any = raw
              if (field.values?.length) {
                const sample = field.values[0]
                if (typeof sample === 'number') parsed = Number(raw)
                else if (typeof sample === 'boolean') parsed = raw === 'true'
              }
              updateParam(key, parsed)
            }}
          >
            <option value="" disabled>{`选择 ${label}`}</option>
            {field.values?.map((value: any) => (
              <option key={String(value)} value={String(value)}>
                {String(value)}
              </option>
            ))}
          </select>
        ) : (
          <input
            type="number"
            step={field.type === 'int' ? field.step || 1 : 'any'}
            className="input"
            style={{ borderColor: isError ? 'var(--danger-color)' : undefined }}
            value={params[key] ?? ''}
            onChange={(event) => {
              const raw = event.target.value
              updateParam(key, field.type === 'int' ? parseInt(raw, 10) : parseFloat(raw))
            }}
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
          <span>{Object.keys(schema).length} 项</span>
        </div>
      </div>

      <div className="parameter-scroll">
        {validationErrors.general && <div className="text-danger p-2">{validationErrors.general}</div>}

        <div className="parameter-model-row">
          <label className="param-label">
            <span>模型路径</span>
            <span className="param-helper">支持本地 `.pt` 权重或项目内模型名</span>
          </label>
          <input className="input" value={model} onChange={(event) => setModel(event.target.value)} />
        </div>

        <div className="parameter-groups">
          {groups.map((group) => {
            const isOpen = expanded[group.id]
            return (
              <section key={group.id} className="param-group-card">
                <button type="button" className="param-group-toggle" onClick={() => toggleGroup(group.id)}>
                  <div>
                    <div className="param-group-title">{group.title}</div>
                    <div className="param-group-description">{group.description}</div>
                  </div>
                  {isOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                </button>
                {isOpen && (
                  <div className="param-grid">
                    {group.keys.map((key) => renderField(key))}
                  </div>
                )}
              </section>
            )
          })}
        </div>
      </div>

      <div className="parameter-footer">
        <div className="flex-col gap-2 mb-4">
          <label className="param-label">
            <span>备注</span>
            <span className="param-helper">可选，用于记录这次调参的目的</span>
          </label>
          <input className="input" value={note} onChange={(event) => setNote(event.target.value)} />
        </div>
        <div className="flex gap-2">
          {onClose && <button className="btn flex-1" onClick={onClose} disabled={validating || loading}>关闭</button>}
          <button className="btn btn-primary flex-1" onClick={handleRun} disabled={loading || validating}>
            {loading ? '正在启动...' : '开始训练（自动校验）'}
          </button>
        </div>
      </div>
    </div>
  )
}
