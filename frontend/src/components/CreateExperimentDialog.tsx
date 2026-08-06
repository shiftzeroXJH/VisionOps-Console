import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'

interface Props {
  existingProjects: string[]
  onClose: () => void
  onCreated: (id: string) => void
}

const TASK_TYPES = [
  { value: 'detection', label: '目标检测 (Detection)' },
  { value: 'segment', label: '实例分割 (Segment)' },
  { value: 'obb', label: '旋转框 (OBB)' },
]

export function CreateExperimentDialog({ existingProjects, onClose, onCreated }: Props) {
  const [form, setForm] = useState({
    description: '',
    project: '',
    task_type: 'detection',
    dataset_root: '',
    pretrained: 'yolo26n.pt',
    save_root: 'runs',
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

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
        pretrained: form.pretrained,
        save_root: form.save_root,
      }
      const res = await api.createExperiment(payload)
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
      style={{
        position: 'fixed', inset: 0,
        backgroundColor: 'rgba(0,0,0,0.5)',
        backdropFilter: 'blur(2px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 100,
      }}
      onClick={(e) => { if (e.target === e.currentTarget && !loading) onClose() }}
    >
      <div className="card" style={{ width: '500px', maxWidth: '100%' }}>
        <h2 style={{ marginBottom: '1rem', fontSize: '1.25rem' }}>创建实验</h2>
        {error && <div className="p-4" style={{ backgroundColor: 'var(--danger-color)', color: '#fff', borderRadius: 'var(--radius-sm)', marginBottom: '1rem' }}>{error}</div>}
        <form className="flex-col gap-4" onSubmit={handleSubmit}>
          <div className="flex-col gap-2">
            <label style={{ fontSize: '0.875rem' }}>实验描述</label>
            <input required className="input" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} placeholder="例如：金具 224 baseline" />
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
              <select className="input" value={form.task_type} onChange={(e) => setForm({ ...form, task_type: e.target.value })}>
                {TASK_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>{t.label}</option>
                ))}
              </select>
            </div>
            <div className="flex-col gap-2 w-full">
              <label style={{ fontSize: '0.875rem' }}>数据集目录 (Dataset Root)</label>
              <input required className="input" value={form.dataset_root} onChange={(e) => setForm({ ...form, dataset_root: e.target.value })} placeholder="C:/datasets/my_dataset" />
            </div>
          </div>

          <div className="flex gap-4">
            <div className="flex-col gap-2 w-full">
              <label style={{ fontSize: '0.875rem' }}>保存目录 (Save Root)</label>
              <input required className="input" value={form.save_root} onChange={(e) => setForm({ ...form, save_root: e.target.value })} placeholder="runs" />
            </div>
            <div className="flex-col gap-2 w-full">
              <label style={{ fontSize: '0.875rem' }}>初始模型 (Model)</label>
              <input required className="input" value={form.pretrained} onChange={(e) => setForm({ ...form, pretrained: e.target.value })} />
            </div>
          </div>

          <div className="flex justify-end gap-2 mt-4 pt-4" style={{ borderTop: '1px solid var(--panel-border)' }}>
            <button type="button" className="btn" onClick={onClose}>取消</button>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? '正在创建...' : '创建实验'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
