import { type Experiment } from '../api'
import { api } from '../api'
import { clsx } from 'clsx'
import { ChevronDown, ChevronRight, Search, Settings, Trash2 } from 'lucide-react'
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
  NOT_STARTED: '未开始',
  QUEUED: '排队中',
  TRAINING: '训练中',
  COMPLETED: '训练完成',
  INTERRUPTED_OR_FAILED: '任务中断/失败',
}

const normalize = (value: unknown) => String(value ?? '').trim().toLocaleLowerCase()

const formatDate = (value: string | undefined) => {
  if (!value) return '-'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString('zh-CN')
}

const projectName = (experiment: Experiment) => {
  const project = String(experiment.project || '').trim()
  return project || UNGROUPED_PROJECT
}

const latestTrainingTime = (experiment: Experiment) => {
  const timestamp = Date.parse(experiment.latest_trial?.created_at || '')
  return Number.isFinite(timestamp) ? timestamp : 0
}

const compareExperiments = (left: Experiment, right: Experiment) => {
  const timeDifference = latestTrainingTime(right) - latestTrainingTime(left)
  if (timeDifference !== 0) return timeDifference
  return left.description.localeCompare(right.description, 'zh-Hans-CN')
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
  const [settingsProject, setSettingsProject] = useState<string | null>(null)

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

    return Array.from(groups.entries())
      .map(([project, items]) => ({
        project,
        experiments: [...items].sort(compareExperiments),
      }))
      .sort((left, right) => {
        const leftLatest = Math.max(0, ...left.experiments.map(latestTrainingTime))
        const rightLatest = Math.max(0, ...right.experiments.map(latestTrainingTime))
        if (rightLatest !== leftLatest) return rightLatest - leftLatest
        return left.project.localeCompare(right.project, 'zh-Hans-CN')
      })
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
                  <button type="button" className="experiment-project-name" onClick={() => toggleProject(group.project)} title={isOpen ? '折叠项目' : '展开项目'}>
                    {group.project}
                  </button>
                  <span className="experiment-project-count">{group.experiments.length}</span>
                  <button type="button" className="btn experiment-project-action" onClick={() => setSettingsProject(group.project)} title="项目设置">
                    <Settings size={13} />
                  </button>
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
                    </div>

                    <div className="experiment-meta">
                      <span>{formatDate(exp.latest_trial?.created_at)}</span>
                      <span>类型: {exp.task_type}</span>
                    </div>

                    <div className="experiment-metrics">
                      <div className="metric-row">
                        <span>状态</span>
                        <span style={{ color: 'var(--text-main)' }}>{statusTextMap[exp.status] || exp.status}</span>
                      </div>
                      {exp.best_metric && (
                        <div className="metric-row" style={{ marginTop: '0.25rem' }}>
                          <span>当前最优 mAP50-95</span>
                          <span style={{ color: 'var(--text-main)' }}>
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
      {settingsProject && (
        <ProjectSettingsDialog
          project={settingsProject}
          experiments={experiments.filter((experiment) => projectName(experiment) === settingsProject)}
          onClose={() => setSettingsProject(null)}
          onSaved={(nextProject) => {
            setExpandedProjects((current) => {
              const next = { ...current, [nextProject]: true }
              if (nextProject !== settingsProject) delete next[settingsProject]
              return next
            })
            setSettingsProject(null)
            onExperimentUpdated?.()
          }}
          onDeleted={() => {
            setSettingsProject(null)
            onExperimentUpdated?.()
          }}
        />
      )}
    </div>
  )
}

function ProjectSettingsDialog({
  project,
  experiments,
  onClose,
  onSaved,
  onDeleted,
}: {
  project: string
  experiments: Experiment[]
  onClose: () => void
  onSaved: (project: string) => void
  onDeleted: () => void
}) {
  const [name, setName] = useState(project)
  const [defaultExportDir, setDefaultExportDir] = useState(experiments[0]?.default_export_dir || 'exports')
  const [confirmation, setConfirmation] = useState('')
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    api.getProjectSettings(project)
      .then((settings) => {
        if (cancelled) return
        setName(settings.project || project)
        setDefaultExportDir(settings.default_export_dir || 'exports')
      })
      .catch((err: any) => {
        if (!cancelled) alert(err?.detail?.error || '加载项目设置失败')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [project])

  const save = async () => {
    const nextName = name.trim()
    if (!nextName) {
      alert('项目名称不能为空')
      return
    }
    setSaving(true)
    try {
      const result = await api.updateProjectSettings(project, {
        name: nextName,
        default_export_dir: defaultExportDir.trim(),
      })
      onSaved(result.project || nextName)
    } catch (err: any) {
      alert(err?.detail?.error || '保存项目设置失败')
    } finally {
      setSaving(false)
    }
  }

  const deleteProject = async () => {
    if (confirmation !== '确认删除') {
      alert('请输入“确认删除”后再删除项目')
      return
    }
    setDeleting(true)
    try {
      await api.deleteProject(project, confirmation)
      onDeleted()
    } catch (err: any) {
      alert(err?.detail?.error || '删除项目失败')
    } finally {
      setDeleting(false)
    }
  }

  const busy = loading || saving || deleting

  return (
    <>
      <div className="dialog-overlay" onClick={(event) => { if (event.target === event.currentTarget && !busy) onClose() }}>
        <div className="card dialog-card settings-dialog">
          <div className="settings-header">
            <div>
              <div className="settings-kicker">PROJECT</div>
              <h2>项目设置</h2>
            </div>
            <button className="btn" onClick={onClose} disabled={busy}>关闭</button>
          </div>

          <div className="settings-section settings-section-stack">
            <label className="settings-form-block">
              <span>项目名称</span>
              <input className="input" value={name} onChange={(event) => setName(event.target.value)} disabled={busy} />
            </label>
            <label className="settings-form-block">
              <span>默认模型导出路径</span>
              <input
                className="input"
                value={defaultExportDir}
                onChange={(event) => setDefaultExportDir(event.target.value)}
                placeholder="exports"
                disabled={busy}
              />
            </label>
            <div className="text-muted" style={{ fontSize: '0.8rem' }}>该项目下共有 {experiments.length} 个任务。</div>
            <div className="flex justify-end gap-2">
              <button className="btn btn-danger" onClick={() => setShowDeleteConfirm(true)} disabled={busy}>
                <Trash2 size={16} /> 删除项目
              </button>
              <button className="btn btn-primary" onClick={save} disabled={busy}>
                {saving ? '保存中...' : '保存设置'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {showDeleteConfirm && (
        <div className="dialog-overlay" onClick={(event) => { if (event.target === event.currentTarget && !deleting) setShowDeleteConfirm(false) }}>
          <div className="card dialog-card settings-dialog" style={{ borderColor: 'rgba(239,68,68,0.45)' }}>
            <div className="settings-header">
              <div>
                <div className="settings-kicker">DANGER</div>
                <h2 className="text-danger">删除项目</h2>
              </div>
              <button className="btn" onClick={() => setShowDeleteConfirm(false)} disabled={deleting}>关闭</button>
            </div>
            <p className="text-muted" style={{ fontSize: '0.9rem' }}>
              会删除“{project}”下所有任务记录、Trial 记录和训练结果 runs 文件。
            </p>
            <label className="settings-form-block">
              <span>输入“确认删除”</span>
              <input
                className="input"
                value={confirmation}
                onChange={(event) => setConfirmation(event.target.value)}
                disabled={deleting}
                autoFocus
              />
            </label>
            <div className="flex justify-end gap-2">
              <button className="btn" onClick={() => setShowDeleteConfirm(false)} disabled={deleting}>取消</button>
              <button className="btn btn-danger" onClick={deleteProject} disabled={deleting || confirmation !== '确认删除'}>
                {deleting ? '删除中...' : '确定删除'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
