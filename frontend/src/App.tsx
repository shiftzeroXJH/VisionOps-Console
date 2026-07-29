import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Experiment } from './api'
import { ExperimentList } from './components/ExperimentList'
import { Workspace } from './components/Workspace'
import { CreateExperimentDialog } from './components/CreateExperimentDialog'
import { Settings, ActivitySquare, Home, Plus } from 'lucide-react'
import { SettingsDialog } from './components/SettingsDialog'
import { HomePage } from './components/HomePage'
import { ModelWorkbench } from './components/ModelWorkbench'

function TrainingPlatform() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [activeExperimentId, setActiveExperimentId] = useState<string | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
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

  useEffect(() => {
    loadExperiments()
  }, [loadExperiments])

  return (
    <div className="app-shell">
      <div className="sidebar-shell">
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <ActivitySquare size={20} />
            <span>YOLO 实验面板</span>
          </div>
          <div className="flex gap-2">
            <button className="btn" style={{ padding: '0.25rem 0.5rem' }} onClick={() => { window.location.hash = '#/' }} title="返回首页">
              <Home size={16} />
            </button>
            <button className="btn" style={{ padding: '0.25rem 0.5rem' }} onClick={() => setShowSettings(true)} title="设置">
              <Settings size={16} />
            </button>
            <button className="btn btn-primary" style={{ padding: '0.25rem 0.5rem' }} onClick={() => setShowCreate(true)} title="创建实验">
              <Plus size={16} />
            </button>
          </div>
        </div>
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
