import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Experiment, type TrainingTaskList } from './api'
import { ExperimentList } from './components/ExperimentList'
import { Workspace } from './components/Workspace'
import { CreateExperimentDialog } from './components/CreateExperimentDialog'
import { Settings, ActivitySquare, Home, ListTodo, Plus } from 'lucide-react'
import { SettingsDialog } from './components/SettingsDialog'
import { HomePage } from './components/HomePage'
import { ModelWorkbench } from './components/ModelWorkbench'
import { TrainingTaskPopover } from './components/TrainingTaskPopover'

const EMPTY_TRAINING_TASKS: TrainingTaskList = {
  groups: [],
  max_parallel_training_tasks: 1,
  running_count: 0,
  queued_count: 0,
  running: [],
  queued: [],
}

function TrainingPlatform() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [activeExperimentId, setActiveExperimentId] = useState<string | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [showTrainingTasks, setShowTrainingTasks] = useState(false)
  const [trainingTasks, setTrainingTasks] = useState<TrainingTaskList>(EMPTY_TRAINING_TASKS)
  const [loading, setLoading] = useState(true)

  const activeIdRef = useRef(activeExperimentId)
  activeIdRef.current = activeExperimentId

  const loadExperiments = useCallback(async () => {
    try {
      const data = await api.getExperiments()
      const nextExperiments = data.experiments || []
      setExperiments(nextExperiments)
      if (nextExperiments.length === 0) {
        setActiveExperimentId(null)
      } else if (!activeIdRef.current || !nextExperiments.some((experiment) => experiment.experiment_id === activeIdRef.current)) {
        setActiveExperimentId(nextExperiments[0].experiment_id)
      }
    } catch (err) {
      console.error('Error loading experiments:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  const loadTrainingTasks = useCallback(async () => {
    try {
      setTrainingTasks(await api.getTrainingTasks())
    } catch (err) {
      console.error('Error loading training tasks:', err)
      throw err
    }
  }, [])

  const refreshTrainingState = useCallback(async () => {
    await Promise.all([loadExperiments(), loadTrainingTasks()])
  }, [loadExperiments, loadTrainingTasks])

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    const poll = async () => {
      await refreshTrainingState().catch(() => {})
      if (!cancelled) {
        timer = window.setTimeout(poll, 60000)
      }
    }
    const handleVisibilityChange = () => {
      if (!document.hidden) void refreshTrainingState().catch(() => {})
    }
    void poll()
    document.addEventListener('visibilitychange', handleVisibilityChange)
    window.addEventListener('training-tasks-changed', handleVisibilityChange)
    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      window.removeEventListener('training-tasks-changed', handleVisibilityChange)
    }
  }, [refreshTrainingState])

  return (
    <div className="app-shell">
      <div className="sidebar-shell">
        <div className="sidebar-header">
          <div className="sidebar-brand" title="YOLO 实验平台" aria-label="YOLO 实验平台">
            <ActivitySquare size={20} />
            <span>YOLO</span>
          </div>
          <div className="sidebar-primary-actions">
            <button className="btn" style={{ padding: '0.25rem 0.5rem' }} onClick={() => { window.location.hash = '#/' }} title="返回首页">
              <Home size={16} />
            </button>
            <button className="btn" style={{ padding: '0.25rem 0.5rem' }} onClick={() => { setShowTrainingTasks(false); setShowSettings(true) }} title="设置">
              <Settings size={16} />
            </button>
            <button
              className="btn training-queue-trigger"
              style={{ padding: '0.25rem 0.5rem' }}
              onClick={() => { setShowTrainingTasks((visible) => !visible); void loadTrainingTasks().catch(() => {}) }}
              aria-expanded={showTrainingTasks}
              title="模型训练列表"
            >
              <ListTodo size={16} />
              {((trainingTasks.total_running_count ?? trainingTasks.running_count) > 0 || (trainingTasks.total_queued_count ?? trainingTasks.queued_count) > 0) && (
                <span className="training-queue-badge">{(trainingTasks.total_running_count ?? trainingTasks.running_count)}/{(trainingTasks.total_queued_count ?? trainingTasks.queued_count)}</span>
              )}
            </button>
            <button className="btn btn-primary" style={{ padding: '0.25rem 0.5rem' }} onClick={() => { setShowTrainingTasks(false); setShowCreate(true) }} title="创建实验">
              <Plus size={16} />
            </button>
          </div>
        </div>
        {showTrainingTasks && (
          <TrainingTaskPopover
            data={trainingTasks}
            onClose={() => setShowTrainingTasks(false)}
            onChanged={refreshTrainingState}
            onSelectExperiment={(experimentId) => {
              setActiveExperimentId(experimentId)
              setShowTrainingTasks(false)
            }}
          />
        )}
        <div className="sidebar-scroll">
          <ExperimentList
            experiments={experiments}
            activeId={activeExperimentId}
            onSelect={setActiveExperimentId}
            onExperimentUpdated={loadExperiments}
          />
        </div>
      </div>

      <div className="main-shell">
        {activeExperimentId ? (
          <Workspace
            experimentId={activeExperimentId}
            onExperimentUpdated={loadExperiments}
            onDeleted={() => {
              setActiveExperimentId(null)
              loadExperiments()
            }}
          />
        ) : (
          <div className="flex items-center" style={{ justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
            {loading ? '正在加载工作台...' : '当前未选中实验'}
          </div>
        )}
      </div>

      {showCreate && (
        <CreateExperimentDialog
          existingProjects={Array.from(new Set(experiments.map((experiment) => experiment.project).filter(Boolean)))}
          onClose={() => setShowCreate(false)}
          onCreated={(id) => {
            setShowCreate(false)
            loadExperiments()
            setActiveExperimentId(id)
          }}
        />
      )}

      {showSettings && <SettingsDialog onClose={() => setShowSettings(false)} />}
    </div>
  )
}

function App() {
  const [route, setRoute] = useState(window.location.hash || '#/')

  useEffect(() => {
    const updateRoute = () => setRoute(window.location.hash || '#/')
    window.addEventListener('hashchange', updateRoute)
    return () => window.removeEventListener('hashchange', updateRoute)
  }, [])

  if (route.startsWith('#/training')) return <TrainingPlatform />
  if (route.startsWith('#/workbench/evaluation')) return <ModelWorkbench key={route.includes('?') ? route : 'workbench'} tab="evaluation" />
  if (route.startsWith('#/workbench')) return <ModelWorkbench key={route.includes('?') ? route : 'workbench'} tab="inference" />
  return <HomePage />
}

export default App
