import { useMemo, useRef, useState } from 'react'
import { Check, ChevronDown, Search } from 'lucide-react'
import type { WorkbenchModel } from '../api'

export type ModelSelection = {
  model_source: 'platform' | 'local'
  trial_id: string
  checkpoint_name: string
  model_path: string
}

interface Props {
  value: ModelSelection
  models: WorkbenchModel[]
  disabled?: boolean
  onChange: (value: ModelSelection) => void
}

const modelLabel = (model: WorkbenchModel) => `${model.project} / ${model.experiment_name} / ${model.trial_name}`

export function ModelSelector({ value, models, disabled, onChange }: Props) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [popupPosition, setPopupPosition] = useState({ left: 0, top: 0, width: 360 })
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const selected = models.find((model) => model.trial_id === value.trial_id)
  const normalizedSearch = search.trim().toLocaleLowerCase()
  const sortedModels = useMemo(() => {
    const experimentOrder = new Map<string, number>()
    models.forEach((model) => {
      if (!experimentOrder.has(model.experiment_id)) experimentOrder.set(model.experiment_id, experimentOrder.size)
    })
    return models.map((model, index) => ({ model, index })).sort((left, right) => {
      const groupDifference = (experimentOrder.get(left.model.experiment_id) || 0) - (experimentOrder.get(right.model.experiment_id) || 0)
      if (groupDifference) return groupDifference
      const timeDifference = String(right.model.created_at || '').localeCompare(String(left.model.created_at || ''))
      return timeDifference || left.index - right.index
    }).map((item) => item.model)
  }, [models])
  const filteredModels = useMemo(
    () => sortedModels.filter((model) => !normalizedSearch || modelLabel(model).toLocaleLowerCase().includes(normalizedSearch)),
    [sortedModels, normalizedSearch],
  )

  const selectModel = (model: WorkbenchModel) => {
    onChange({
      ...value,
      trial_id: model.trial_id,
      checkpoint_name: model.default_checkpoint || model.checkpoints[0]?.name || '',
    })
    setSearch('')
    setOpen(false)
  }

  const openOptions = () => {
    const rect = wrapperRef.current?.getBoundingClientRect()
    if (rect) setPopupPosition({ left: rect.left, top: rect.bottom + 4, width: rect.width })
    setOpen(true)
  }

  return (
    <div className="model-selector">
      <div className="segmented-control" aria-label="模型来源">
        <button className={value.model_source === 'platform' ? 'active' : ''} disabled={disabled} onClick={() => onChange({ ...value, model_source: 'platform' })}>训练平台</button>
        <button className={value.model_source === 'local' ? 'active' : ''} disabled={disabled} onClick={() => onChange({ ...value, model_source: 'local' })}>本地路径</button>
      </div>
      {value.model_source === 'platform' ? (
        <>
          <div
            className={`model-combobox ${open ? 'open' : ''}`}
            ref={wrapperRef}
            onBlur={(event) => {
              if (!wrapperRef.current?.contains(event.relatedTarget as Node | null)) {
                setOpen(false)
                setSearch('')
              }
            }}
          >
            <Search size={15} />
            <input
              value={open ? search : selected ? modelLabel(selected) : ''}
              disabled={disabled}
              placeholder="选择训练模型"
              aria-label="搜索训练模型"
              onFocus={() => { openOptions(); setSearch('') }}
              onChange={(event) => { setSearch(event.target.value); openOptions() }}
            />
            <ChevronDown size={15} />
            {open && (
              <div className="model-options" role="listbox" aria-label="训练模型列表" style={popupPosition}>
                {filteredModels.map((model) => (
                  <button key={model.trial_id} type="button" role="option" aria-selected={model.trial_id === value.trial_id} onMouseDown={(event) => event.preventDefault()} onClick={() => selectModel(model)}>
                    <span>{model.trial_name}</span>
                    <small>{model.project} / {model.experiment_name}</small>
                    {model.trial_id === value.trial_id && <Check size={15} />}
                  </button>
                ))}
                {!filteredModels.length && <div className="model-options-empty">没有匹配的训练模型</div>}
              </div>
            )}
          </div>
          <select
            className="input checkpoint-select"
            aria-label="Checkpoint"
            value={selected ? value.checkpoint_name || selected.default_checkpoint : ''}
            disabled={disabled || !selected}
            onChange={(event) => onChange({ ...value, checkpoint_name: event.target.value })}
          >
            {!selected && <option value="" disabled>选择 checkpoint</option>}
            {(selected?.checkpoints || []).map((checkpoint) => (
              <option key={checkpoint.name} value={checkpoint.name}>{checkpoint.label}</option>
            ))}
          </select>
        </>
      ) : (
        <input className="input model-input" value={value.model_path} disabled={disabled} placeholder="D:\models\best.pt 或 model.onnx" onChange={(event) => onChange({ ...value, model_path: event.target.value })} />
      )}
    </div>
  )
}
