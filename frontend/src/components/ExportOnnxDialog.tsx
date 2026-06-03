import { useMemo, useState } from 'react'
import { api } from '../api'

interface Props {
  trialId: string
  modelStem: string
  imgsz: number
  architecture: string
  onClose: () => void
}

const DEFAULT_OUTPUT_DIR = 'C:\\Users\\Administrator\\Downloads'
const ILLEGAL_NAME_PATTERN = /[<>:"/\\|?*\u0000-\u001f]/

export function ExportOnnxDialog({ trialId, modelStem, imgsz, architecture, onClose }: Props) {
  const [modelName, setModelName] = useState(modelStem)
  const [outputDir, setOutputDir] = useState(DEFAULT_OUTPUT_DIR)
  const [submitting, setSubmitting] = useState(false)

  const validationError = useMemo(() => {
    const trimmed = modelName.trim()
    if (!trimmed) return '模型名字不能为空'
    if (ILLEGAL_NAME_PATTERN.test(trimmed)) return '模型名字包含 Windows 非法字符'
    if (/[. ]$/.test(trimmed)) return '模型名字不能以空格或点结尾'
    if (!outputDir.trim()) return '导出目录不能为空'
    if (!Number.isFinite(imgsz) || imgsz <= 0) return '当前 Trial 缺少有效的 imgsz，无法导出'
    return ''
  }, [imgsz, modelName, outputDir])

  const previewName = useMemo(() => {
    const normalized = modelName.trim() || '模型名字'
    return `${normalized}-${architecture}-${imgsz}.onnx`
  }, [architecture, imgsz, modelName])

  const handleExport = async () => {
    if (validationError) {
      alert(validationError)
      return
    }
    setSubmitting(true)
    try {
      const result = await api.exportTrialOnnx(trialId, {
        model_name: modelName.trim(),
        output_dir: outputDir.trim(),
      })
      alert(`导出完成：\n${result.output_path}`)
      onClose()
    } catch (err: any) {
      alert(err?.detail?.error || '导出 ONNX 失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div
      className="dialog-overlay"
      onClick={(event) => {
        if (event.target === event.currentTarget && !submitting) onClose()
      }}
    >
      <div className="card dialog-card">
        <h2 style={{ marginBottom: '1rem', fontSize: '1.1rem' }}>导出 ONNX</h2>
        <div className="flex-col gap-4">
          <label className="flex-col gap-1">
            <span>模型名字</span>
            <input
              className="input"
              value={modelName}
              onChange={(event) => setModelName(event.target.value)}
              placeholder="请输入模型名字"
              disabled={submitting}
              autoFocus
            />
          </label>
          <label className="flex-col gap-1">
            <span>导出目录</span>
            <input
              className="input"
              value={outputDir}
              onChange={(event) => setOutputDir(event.target.value)}
              placeholder={DEFAULT_OUTPUT_DIR}
              disabled={submitting}
            />
          </label>
          <div className="card" style={{ padding: '0.75rem', backgroundColor: 'rgba(255,255,255,0.45)' }}>
            <div className="text-muted" style={{ fontSize: '0.75rem' }}>文件名预览</div>
            <div style={{ marginTop: '0.25rem', wordBreak: 'break-all', fontFamily: 'monospace' }}>{previewName}</div>
          </div>
          {validationError && <div className="text-danger" style={{ fontSize: '0.85rem' }}>{validationError}</div>}
        </div>
        <div className="flex justify-end gap-2 pt-4" style={{ marginTop: '1rem', borderTop: '1px solid var(--panel-border)' }}>
          <button className="btn" onClick={onClose} disabled={submitting}>取消</button>
          <button className="btn btn-primary" onClick={handleExport} disabled={submitting}>
            {submitting ? '导出中...' : '开始导出'}
          </button>
        </div>
      </div>
    </div>
  )
}
