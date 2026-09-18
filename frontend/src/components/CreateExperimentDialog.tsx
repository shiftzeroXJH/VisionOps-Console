import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'

interface Props {
  existingProjects: string[]
  onClose: () => void
  onCreated: (id: string) => void
}

type Catalog = { tasks: { value: string; label: string }[]; models: { task_type: string; model_family: string; model_scale: string; filename: string }[] }

export function CreateExperimentDialog({ existingProjects, onClose, onCreated }: Props) {
  const [form, setForm] = useState({
    description: '',
    project: '',
    task_type: 'detection',
    dataset_root: '',
    model_family: '26',
    model_scale: 'n',
    dataset_yaml: '',
    save_root: 'runs',
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [catalog, setCatalog] = useState<Catalog>({ tasks: [], models: [] })
  const [yamlCandidates, setYamlCandidates] = useState<string[]>([])
  useEffect(() => {
    fetch('/api/model-catalog').then(async response => {
      if (!response.ok) throw new Error('模型目录加载失败')
      setCatalog(await response.json())
    }).catch(error => setError(error.message))
  }, [])
  const taskModels = catalog.models.filter(model => model.task_type === form.task_type)
  const families = [...new Set(taskModels.map(model => model.model_family))]
  const scales = taskModels.filter(model => model.model_family === form.model_family)
  const selectedModel = scales.find(model => model.model_scale === form.model_scale)

  const projectOptions = Array.from(new Set(existingProjects.filter(Boolean))).sort((a, b) => a.localeCompare(b, 'zh-Hans-CN'))
  const fallbackProject = (description: string) => {
    const normalized = description.trim()
    return normalized ? Array.from(normalized).slice(0, 2).join('') : '未分组'
  }

  const formatCreateExperimentError = (res: any) => {
    if (!res) return '创建实验失败'
    if (res.status === 'needs_dataset_yaml') {
      const candidates = Array.isArray(res.yaml_candidates) ? res.yaml_candidates : []
      if (candidates.length === 0) {
        return `${res.message || '未找到可用的数据集 YAML'}。请检查 Dataset Root 是否正确，或补充 dataset yaml。`
      }
      return `${res.message || '需要明确指定 dataset yaml'}。候选文件：${candidates.join('，')}`
    }
    return res.message || res.detail?.error || '创建实验失败'
  }

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (event.key === 'Escape' && !loading) onClose()
  }, [loading, onClose])

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const payload = {
        description: form.description,
        project: form.project.trim() || fallbackProject(form.description),
        task_type: form.task_type,
        dataset_root: form.dataset_root,
        model_family: form.model_family,
        model_scale: form.model_scale,
        dataset_yaml: form.dataset_yaml || undefined,
        save_root: form.save_root,
      }
      const res = await api.createExperiment(payload)
      if (res.status === 'needs_dataset_yaml') setYamlCandidates(res.yaml_candidates || [])
      if (res.experiment_id) {
        onCreated(res.experiment_id)
        return
      }
      setError(formatCreateExperimentError(res))
    } catch (err: any) {
      setError(formatCreateExperimentError(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      className="dialog-overlay"
      onClick={(e) => { if (e.target === e.currentTarget && !loading) onClose() }}
    >
      <div className="card dialog-card" style={{ width: '480px' }}>
        <h2 style={{ marginBottom: '1rem', fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-primary)' }}>创建实验</h2>
        {error && <div className="p-4" style={{ backgroundColor: 'var(--danger-color)', color: '#fff', borderRadius: 'var(--radius-sm)', marginBottom: '1rem' }}>{error}</div>}
        <form className="flex-col gap-4" onSubmit={handleSubmit}>
          <div className="flex-col gap-2">
            <label style={{ fontSize: '0.875rem' }}>实验描述</label>
            <input required className="input" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} placeholder="例如：零件检测 224 baseline" />
          </div>

          <div className="flex-col gap-2">
            <label style={{ fontSize: '0.875rem' }}>项目</label>
            <input
              className="input"
              list="project-options"
              value={form.project}
              onChange={(e) => setForm({ ...form, project: e.target.value })}
              placeholder={`留空则使用：${fallbackProject(form.description)}`}
            />
            <datalist id="project-options">
              {projectOptions.map((project) => (
                <option key={project} value={project} />
              ))}
            </datalist>
          </div>

          <div className="flex gap-4">
            <div className="flex-col gap-2 w-full">
              <label style={{ fontSize: '0.875rem' }}>任务类型 (Task Type)</label>
              <select className="input" value={form.task_type} onChange={(e) => {
                const task_type = e.target.value
                const compatible = catalog.models.some(model => model.task_type === task_type && model.model_family === form.model_family)
                setForm({ ...form, task_type, model_family: compatible ? form.model_family : '26' })
              }}>
                {catalog.tasks.map((t) => (
                  <option key={t.value} value={t.value}>{t.label}</option>
                ))}
              </select>
            </div>
            <div className="flex-col gap-2 w-full">
              <label style={{ fontSize: '0.875rem' }}>数据集目录 (Dataset Root)</label>
              <input required className="input" value={form.dataset_root} onChange={(e) => { setForm({ ...form, dataset_root: e.target.value, dataset_yaml: '' }); setYamlCandidates([]) }} placeholder="C:/datasets/my_dataset" />
            </div>
          </div>

          {yamlCandidates.length > 0 && <label>数据集 YAML<select required className="input" value={form.dataset_yaml} onChange={e => setForm({ ...form, dataset_yaml: e.target.value })}><option value="">选择 YAML</option>{yamlCandidates.map(path => <option key={path} value={path}>{path}</option>)}</select></label>}
          <div className="flex gap-4">
            <label className="flex-col gap-2 w-full">YOLO 系列<select className="input" value={form.model_family} onChange={e => setForm({ ...form, model_family: e.target.value })}>{families.map(family => <option key={family} value={family}>{family === 'v8' ? 'YOLOv8' : `YOLO${family}`}</option>)}</select></label>
            <label className="flex-col gap-2 w-full">模型规格<select className="input" value={form.model_scale} onChange={e => setForm({ ...form, model_scale: e.target.value })}>{scales.map(model => <option key={model.model_scale} value={model.model_scale}>{model.model_scale}</option>)}</select></label>
          </div>
          <div className="flex gap-4">
            <div className="flex-col gap-2 w-full">
              <label style={{ fontSize: '0.875rem' }}>保存目录 (Save Root)</label>
              <input required className="input" value={form.save_root} onChange={(e) => setForm({ ...form, save_root: e.target.value })} placeholder="runs" />
            </div>
            <div className="flex-col gap-2 w-full">
              <label style={{ fontSize: '0.875rem' }}>初始模型 (Model)</label>
              <output className="input font-mono" style={{ overflowWrap: 'anywhere' }}>{selectedModel?.filename || '-'}</output>
            </div>
          </div>

          <div className="flex justify-end gap-2 mt-4 pt-4" style={{ borderTop: '1px solid var(--panel-border)' }}>
            <button type="button" className="btn" onClick={onClose}>取消</button>
            <button type="submit" className="btn btn-primary" disabled={loading || !selectedModel}>
              {loading ? '正在创建...' : '创建实验'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
