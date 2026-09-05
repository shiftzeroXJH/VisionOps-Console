import { ActivitySquare, ArrowRight, ScanSearch } from 'lucide-react'
import { VisionOpsLogo } from './VisionOpsLogo'

export function HomePage() {
  return (
    <main className="home-page">
      <header className="home-header">
        <div className="home-brand">
          <VisionOpsLogo size={28} />
          <div className="brand-logotype" style={{ fontSize: '1.2rem' }}>
            <span className="brand-logotype-vision">VISION</span>
            <span className="brand-logotype-ops">OPS</span>
            <span className="brand-pulse-dot" />
          </div>
        </div>
        <span>半导体 AOI 工业视觉与训练平台</span>
      </header>
      <section className="home-content">
        <div className="home-title-block">
          <span className="home-kicker">工作空间</span>
          <h1>选择要进入的工作台</h1>
        </div>
        <div className="home-entry-grid">
          <button className="home-entry" onClick={() => { window.location.hash = '#/training' }}>
            <span className="home-entry-icon"><ActivitySquare size={28} /></span>
            <span className="home-entry-copy">
              <strong>训练平台</strong>
              <span>管理实验、训练任务、参数与结果对比</span>
            </span>
            <ArrowRight size={20} />
          </button>
          <button className="home-entry" onClick={() => { window.location.hash = '#/workbench/inference' }}>
            <span className="home-entry-icon home-entry-icon-green"><ScanSearch size={28} /></span>
            <span className="home-entry-copy">
              <strong>模型工作台</strong>
              <span>批量图片推理、模型评估与标注对比</span>
            </span>
            <ArrowRight size={20} />
          </button>
        </div>
      </section>
    </main>
  )
}
