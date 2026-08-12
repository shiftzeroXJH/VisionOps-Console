import { ArrowDown, ArrowUp, ListTodo, Trash2 } from 'lucide-react'
import { api, type TrainingTask, type TrainingTaskList } from '../api'

interface Props {
  data: TrainingTaskList
  onChanged: () => Promise<void>
  onSelectExperiment: (experimentId: string) => void
}

const formatDateTime = (value?: string) => {
  if (!value) return '-'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

const modelName = (value: string) => value.replace(/\\/g, '/').split('/').pop() || value

const errorMessage = (error: unknown, fallback: string) => {
  if (!error || typeof error !== 'object' || !("detail" in error)) return fallback
  const detail = error.detail
  if (!detail || typeof detail !== 'object' || !("error" in detail)) return fallback
  return typeof detail.error === 'string' ? detail.error : fallback
}

function TaskRow({
  task,
  queued,
  first,
  last,
  onChanged,
  onSelectExperiment,
}: {
  task: TrainingTask
  queued: boolean
  first?: boolean
  last?: boolean
  onChanged: () => Promise<void>
  onSelectExperiment: (experimentId: string) => void
}) {
  const move = async (position: number) => {
    try {
      await api.reorderTrainingTask(task.queue_id, position)
      await onChanged()
    } catch (err: unknown) {
      alert(errorMessage(err, '调整顺序失败'))
    }
  }

  const cancel = async () => {
    if (!confirm(`确定取消“${task.experiment_name}”的排队训练吗？`)) return
    try {
      await api.cancelTrainingTask(task.queue_id)
      await onChanged()
    } catch (err: unknown) {
      alert(errorMessage(err, '取消排队失败'))
    }
  }

  return (
    <div className="training-task-row">
      <button className="training-task-main" onClick={() => onSelectExperiment(task.experiment_id)}>
        <span className={`training-task-state ${queued ? 'queued' : 'running'}`}>
          {queued ? `#${task.position} 排队中` : '训练中'}
        </span>
        <strong>{task.experiment_name}</strong>
        <span className="text-muted">
          {task.project} · {task.training_mode === 'continued' ? `续训自 ${task.parent_display_name || task.parent_trial_id}` : modelName(task.model)}
        </span>
        <span className="text-muted">
          {queued ? `提交：${formatDateTime(task.created_at)}` : `开始：${formatDateTime(task.started_at)}`}
        </span>
      </button>
      {queued && (
        <div className="training-task-actions">
          <button className="icon-btn" disabled={first} onClick={() => move(task.position - 1)} title="上移">
            <ArrowUp size={16} />
          </button>
          <button className="icon-btn" disabled={last} onClick={() => move(task.position + 1)} title="下移">
            <ArrowDown size={16} />
          </button>
          <button className="icon-btn danger" onClick={cancel} title="取消排队">
            <Trash2 size={16} />
          </button>
        </div>
      )}
    </div>
  )
}

export function TrainingTaskPopover({ data, onChanged, onSelectExperiment }: Props) {
  return (
    <div className="training-task-popover">
      <div className="training-task-popover-title">
        <ListTodo size={16} />
        <strong>模型训练列表</strong>
      </div>

      <div className="training-capacity">
        <span>并行</span>
        <strong>{data.running_count} / {data.max_parallel_training_tasks}</strong>
        <span className="text-muted">排队 {data.queued_count}</span>
      </div>

      <div className="training-task-popover-scroll">
        {data.running.length > 0 && (
          <section className="training-task-section">
            <h3>训练中</h3>
            {data.running.map((task) => (
              <TaskRow key={task.queue_id} task={task} queued={false} onChanged={onChanged} onSelectExperiment={onSelectExperiment} />
            ))}
          </section>
        )}

        {data.queued.length > 0 && (
          <section className="training-task-section">
            <h3>排队中</h3>
            {data.queued.map((task, index) => (
              <TaskRow
                key={task.queue_id}
                task={task}
                queued
                first={index === 0}
                last={index === data.queued.length - 1}
                onChanged={onChanged}
                onSelectExperiment={onSelectExperiment}
              />
            ))}
          </section>
        )}

        {data.running.length === 0 && data.queued.length === 0 && (
          <div className="training-task-empty">当前没有训练或排队任务</div>
        )}
      </div>
    </div>
  )
}
