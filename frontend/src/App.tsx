import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Experiment } from './api'
import { ExperimentList } from './components/ExperimentList'
import { Workspace } from './components/Workspace'
import { CreateExperimentDialog } from './components/CreateExperimentDialog'
import { Settings, ActivitySquare, Plus } from 'lucide-react'
import { SettingsDialog } from './components/SettingsDialog'

function App() {
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
      setExperiments(data.experiments || [])
      if (data.experiments?.length > 0 && !activeIdRef.current) {
        setActiveExperimentId(data.experiments[0].experiment_id)
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

export default App
