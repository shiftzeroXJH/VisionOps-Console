import { type Experiment } from '../api'
import { api } from '../api'
import { clsx } from 'clsx'
import { Check, ChevronDown, ChevronRight, Edit2, Search, X } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

interface Props {
  experiments: Experiment[]
  activeId: string | null
  onSelect: (id: string) => void
  onExperimentUpdated?: () => void
}

type ExperimentGroup = {
  project: string
  experiments: Experiment[]
}

const UNGROUPED_PROJECT = '未分组'

const statusTextMap: Record<string, string> = {
  COMPLETED: '已完成',
  FAILED: '训练失败',
  CANCELLED: '已取消',
  TRAINING: '训练中',
  ANALYZING: '分析中',
  RETRAINING: '重新训练中',
  WAITING_USER_CONFIRM: '待确认',
  AUTO_TUNE_PENDING: '等待调参',
  READY: '准备就绪',
}

const normalize = (value: unknown) => String(value ?? '').trim().toLocaleLowerCase()

const projectName = (experiment: Experiment) => {
  const project = String(experiment.project || '').trim()
  return project || UNGROUPED_PROJECT
}

const tokenTextForExperiment = (experiment: Experiment) => {
  const statusText = statusTextMap[experiment.status] || experiment.status
  return normalize([
    projectName(experiment),
    experiment.description,
    experiment.experiment_id,
    experiment.task_type,
    experiment.status,
    statusText,
    experiment.pretrained_model,
    experiment.dataset_root,
    experiment.dataset_yaml,
  ].join(' '))
}

export function ExperimentList({ experiments, activeId, onSelect, onExperimentUpdated }: Props) {
  const [query, setQuery] = useState('')
  const [expandedProjects, setExpandedProjects] = useState<Record<string, boolean>>({})
  const [editingProject, setEditingProject] = useState<string | null>(null)
  const [projectValue, setProjectValue] = useState('')
  const [savingProject, setSavingProject] = useState(false)

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'COMPLETED': return 'badge-success'
      case 'FAILED':
      case 'CANCELLED': return 'badge-danger'
      case 'TRAINING':
      case 'ANALYZING':
      case 'RETRAINING':
      case 'WAITING_USER_CONFIRM':
      case 'AUTO_TUNE_PENDING':
        return 'badge-warning'
      default: return ''
    }
  }

  const getStatusText = (status: string) => statusTextMap[status] || status.replace(/_/g, ' ')

  const groupedExperiments = useMemo(() => {
    const words = normalize(query).split(/\s+/).filter(Boolean)
    const groups = new Map<string, Experiment[]>()

    experiments.forEach((experiment) => {
      const project = projectName(experiment)
      const projectMatches = words.length > 0 && words.every((word) => normalize(project).includes(word))
      const experimentMatches = words.length === 0 || projectMatches || words.every((word) => tokenTextForExperiment(experiment).includes(word))
      if (!experimentMatches) return
      groups.set(project, [...(groups.get(project) || []), experiment])
    })

    return Array.from(groups.entries()).map(([project, items]) => ({ project, experiments: items }))
  }, [experiments, query])

  useEffect(() => {
    if (query.trim()) {
      setExpandedProjects(Object.fromEntries(groupedExperiments.map((group) => [group.project, true])))
      return
    }

    const activeExperiment = experiments.find((experiment) => experiment.experiment_id === activeId)
    const defaultProject = activeExperiment ? projectName(activeExperiment) : groupedExperiments[0]?.project
    if (!defaultProject) return
    setExpandedProjects((current) => {
      if (current[defaultProject]) return current
      return { ...current, [defaultProject]: true }
    })
  }, [activeId, experiments, groupedExperiments, query])

  const toggleProject = (project: string) => {
    setExpandedProjects((current) => ({ ...current, [project]: !current[project] }))
  }

  const startProjectEdit = (project: string) => {
    setEditingProject(project)
    setProjectValue(project)
  }

  const cancelProjectEdit = () => {
    setEditingProject(null)
    setProjectValue('')
  }

  const saveProject = async (group: ExperimentGroup) => {
    const nextProject = projectValue.trim()
    if (!nextProject) {
      alert('项目名称不能为空')
      return
    }
    if (nextProject === group.project) {
      cancelProjectEdit()
      return
    }

    setSavingProject(true)
    try {
      await Promise.all(group.experiments.map((experiment) => (
        api.updateExperiment(experiment.experiment_id, { project: nextProject })
      )))
      setExpandedProjects((current) => {
        const next = { ...current, [nextProject]: true }
        delete next[group.project]
        return next
      })
      setEditingProject(null)
      setProjectValue('')
      onExperimentUpdated?.()
    } catch (err: any) {
      alert(err?.detail?.error || '修改项目失败')
    } finally {
      setSavingProject(false)
    }
  }

  if (experiments.length === 0) {
    return (
      <div className="p-4" style={{ color: 'var(--text-muted)', fontSize: '0.875rem', textAlign: 'center' }}>
        暂无实验记录。<br />请点击上方 + 新建实验。
      </div>
    )
  }

  return (
    <div className="experiment-list-panel">
      <div className="experiment-search">
        <Search size={15} />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="搜索任务、项目、模型、状态"
        />
      </div>

      {groupedExperiments.length === 0 ? (
        <div className="p-4" style={{ color: 'var(--text-muted)', fontSize: '0.875rem', textAlign: 'center' }}>没有匹配的任务</div>
      ) : (
        <div className="experiment-list">
          {groupedExperiments.map((group: ExperimentGroup) => {
            const isOpen = Boolean(expandedProjects[group.project])
            return (
              <section key={group.project} className="experiment-project-group">
                <div className="experiment-project-header">
                  <button type="button" className="experiment-project-chevron" onClick={() => toggleProject(group.project)} title={isOpen ? '折叠项目' : '展开项目'}>
                    {isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                  </button>
                  {editingProject === group.project ? (
                    <div className="experiment-project-editor">
                      <input
                        className="input"
                        value={projectValue}
                        onChange={(event) => setProjectValue(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter') saveProject(group)
                          if (event.key === 'Escape') cancelProjectEdit()
                        }}
                        autoFocus
                      />
                      <button className="btn btn-primary experiment-project-action" onClick={() => saveProject(group)} disabled={savingProject} title="保存项目名">
                        <Check size={14} />
                      </button>
                      <button className="btn experiment-project-action" onClick={cancelProjectEdit} disabled={savingProject} title="取消">
                        <X size={14} />
                      </button>
                    </div>
                  ) : (
                    <>
                      <button type="button" className="experiment-project-name" onClick={() => toggleProject(group.project)} title={isOpen ? '折叠项目' : '展开项目'}>
                        {group.project}
                      </button>
                      <span className="experiment-project-count">{group.experiments.length}</span>
                      <button type="button" className="btn experiment-project-action" onClick={() => startProjectEdit(group.project)} title="修改项目名">
                        <Edit2 size={13} />
                      </button>
                    </>
                  )}
                </div>

                {isOpen && group.experiments.map((exp) => (
                  <div
                    key={exp.experiment_id}
                    className={clsx('card experiment-card', { active: exp.experiment_id === activeId })}
                    style={{
                      cursor: 'pointer',
                      transition: 'background-color var(--transition-fast)',
                      borderColor: exp.experiment_id === activeId ? 'var(--primary-color)' : 'var(--panel-border)',
                      backgroundColor: exp.experiment_id === activeId ? 'rgba(59, 130, 246, 0.05)' : undefined,
                    }}
                    onClick={() => onSelect(exp.experiment_id)}
                  >
                    <div className="experiment-card-header">
                      <span className="experiment-title">{exp.description}</span>
                      <span className={clsx('badge', getStatusBadge(exp.status))}>
                        {getStatusText(exp.status)}
                      </span>
                    </div>

                    <div className="experiment-meta">
                      <span>试验次数: {exp.trial_count}</span>
                      <span>类型: {exp.task_type}</span>
                    </div>

                    <div className="experiment-metrics">
                      <div className="metric-row">
                        <span>目标指标 ({exp.goal.metric})</span>
                        <span style={{ color: 'var(--text-main)' }}>{exp.goal.target}</span>
                      </div>
                      {exp.best_metric && (
                        <div className="metric-row" style={{ marginTop: '0.25rem' }}>
                          <span>当前最优</span>
                          <span className={exp.best_metric.value >= exp.goal.target ? 'text-success' : 'text-warning'}>
                            {exp.best_metric.value.toFixed(4)}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </section>
            )
          })}
        </div>
      )}
    </div>
  )
}
