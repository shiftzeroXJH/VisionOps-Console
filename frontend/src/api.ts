export type Experiment = {
  experiment_id: string;
  description: string;
  project: string;
  status: string;
  task_type: string;
  dataset_root: string;
  dataset_yaml?: string;
  pretrained_model: string;
  default_export_dir?: string;
  internal_status?: string;
  trial_count: number;
  best_metric?: {
    trial_id: string;
    iteration: number;
    metric: string;
    value: number;
  };
  latest_trial?: {
    trial_id: string;
    iteration: number;
    status: string;
    internal_status?: string;
    created_at: string;
    [key: string]: unknown;
  };
};

export type TrainingTask = {
  queue_id: string;
  experiment_id: string;
  experiment_name: string;
  project: string;
  task_type: string;
  model: string;
  params: Record<string, unknown>;
  status: 'RUNNING' | 'QUEUED';
  position: number;
  trial_id?: string;
  created_at: string;
  started_at?: string;
  parent_trial_id?: string;
  parent_display_name?: string;
  training_mode?: 'fresh' | 'continued';
};

export type TrainingTaskList = {
  max_parallel_training_tasks: number;
  running_count: number;
  queued_count: number;
  running: TrainingTask[];
  queued: TrainingTask[];
};

export type WorkbenchModel = {
  trial_id: string;
  trial_name: string;
  experiment_id: string;
  experiment_name: string;
  project: string;
  task_type: 'detection' | 'segment' | 'obb' | string;
  created_at?: string;
  path: string;
  default_checkpoint: string;
  checkpoints: Array<{
    name: string;
    label: string;
    epoch?: number | null;
    path: string;
  }>;
};

export type Detection = {
  class_id: number;
  class_name: string;
  confidence?: number | null;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  polygon?: Array<[number, number]>;
};

export type WorkbenchRoi = {
  cx: number;
  cy: number;
  width: number;
  height: number;
  angle: number;
};

export type WorkbenchImage = {
  image_id: string;
  name: string;
  width: number;
  height: number;
  status?: string;
  error?: string;
  detections?: Detection[];
  labels?: Detection[];
  roi?: WorkbenchRoi | null;
  rotation?: number;
  revision?: number;
};

async function safeThrowError(res: Response): Promise<never> {
  let detail: any;
  try {
    detail = await res.json();
  } catch {
    detail = { detail: { error: `HTTP ${res.status}: ${res.statusText}` } };
  }
  throw detail;
}

export const api = {
  async getExperiments(): Promise<{ experiments: Experiment[] }> {
    const res = await fetch('/api/experiments');
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async clearValidationCache() {
    const res = await fetch('/api/settings/clear-validation-cache', {
      method: 'POST'
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getSettings() {
    const res = await fetch('/api/settings');
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async updateSettings(payload: { yolo_python: string; max_parallel_training_tasks: number }) {
    const res = await fetch('/api/settings', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async createExperiment(payload: any) {
    const res = await fetch('/api/experiments', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getRemoteServers() {
    const res = await fetch('/api/remote-servers');
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async createRemoteServer(payload: any) {
    const res = await fetch('/api/remote-servers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async testRemoteServer(remoteServerId: string) {
    const res = await fetch(`/api/remote-servers/${remoteServerId}/test`, {
      method: 'POST'
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getExperiment(experimentId: string) {
    const res = await fetch(`/api/experiments/${experimentId}`);
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async updateExperiment(experimentId: string, payload: { description?: string; project?: string }) {
    const res = await fetch(`/api/experiments/${experimentId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getProjectSettings(project: string) {
    const res = await fetch(`/api/projects/${encodeURIComponent(project)}`);
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async updateProjectSettings(project: string, payload: { name?: string; default_export_dir?: string }) {
    const res = await fetch(`/api/projects/${encodeURIComponent(project)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async deleteProject(project: string, confirmation: string) {
    const res = await fetch(`/api/projects/${encodeURIComponent(project)}`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirmation })
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getComparison(experimentId: string) {
    const res = await fetch(`/api/experiments/${experimentId}/comparison`);
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getParams(experimentId: string) {
    const res = await fetch(`/api/experiments/${experimentId}/params`);
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async validateParams(experimentId: string, payload: any) {
    const res = await fetch(`/api/experiments/${experimentId}/params/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params: payload })
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async runTrial(experimentId: string, payload: any) {
    const res = await fetch(`/api/experiments/${experimentId}/trials/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getTrainingTasks(): Promise<TrainingTaskList> {
    const res = await fetch('/api/training-tasks');
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async cancelTrainingTask(queueId: string) {
    const res = await fetch(`/api/training-tasks/${queueId}/cancel`, { method: 'POST' });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async reorderTrainingTask(queueId: string, position: number) {
    const res = await fetch(`/api/training-tasks/${queueId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ position })
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async registerRemoteTrial(experimentId: string, payload: any) {
    const res = await fetch(`/api/experiments/${experimentId}/trials/remote-register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async importRemoteTrial(experimentId: string, payload: any) {
    const res = await fetch(`/api/experiments/${experimentId}/trials/import-remote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async syncRemoteTrial(trialId: string) {
    const res = await fetch(`/api/trials/${trialId}/remote-sync`, {
      method: 'POST'
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async exportTrialOnnx(trialId: string, payload: { model_name: string; output_dir: string }) {
    const res = await fetch(`/api/trials/${trialId}/export-onnx`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async validateTrialPreview(trialId: string, payload: { image_limit: number; conf: number }) {
    const res = await fetch(`/api/trials/${trialId}/validate-preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getValidationPreview(trialId: string) {
    const res = await fetch(`/api/trials/${trialId}/validation-preview`);
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async importTrial(experimentId: string, payload: any) {
    const res = await fetch(`/api/experiments/${experimentId}/trials/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getJob(jobId: string) {
    const res = await fetch(`/jobs/${jobId}`);
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getTrialSummary(trialId: string) {
    const res = await fetch(`/api/trials/${trialId}/summary`);
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getTrialContinuation(trialId: string) {
    const res = await fetch(`/api/trials/${trialId}/continuation`);
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async continueTrial(trialId: string, payload: {
    additional_epochs: number;
    lr0: number;
    patience: number;
    note?: string;
    enqueue_if_busy?: boolean;
  }) {
    const res = await fetch(`/api/trials/${trialId}/continue`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async renameTrial(trialId: string, payload: { display_name: string }) {
    const res = await fetch(`/api/trials/${trialId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async deleteExperiment(experimentId: string, keepFiles: boolean = true, force: boolean = false) {
    const res = await fetch(`/api/experiments/${experimentId}?keep_files=${keepFiles}&force=${force}`, {
      method: 'DELETE'
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async cancelExperiment(experimentId: string, reason?: string) {
    const res = await fetch(`/api/experiments/${experimentId}/cancel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason })
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async deleteTrial(trialId: string, keepFiles: boolean = true, force: boolean = false) {
    const res = await fetch(`/api/trials/${trialId}?keep_files=${keepFiles}&force=${force}`, {
      method: 'DELETE'
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getExperimentCurves(experimentId: string) {
    const res = await fetch(`/api/experiments/${experimentId}/curves`);
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getTrialVisualizations(trialId: string) {
    const res = await fetch(`/api/trials/${trialId}/visualizations`);
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getWorkbenchModels(): Promise<{ models: WorkbenchModel[]; effective_yolo_python: string }> {
    const res = await fetch('/api/workbench/models');
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async createWorkbenchSession() {
    const res = await fetch('/api/workbench/sessions', { method: 'POST' });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getWorkbenchSession(sessionId: string) {
    const res = await fetch(`/api/workbench/sessions/${sessionId}`);
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async uploadWorkbenchImages(sessionId: string, files: File[]) {
    const images: WorkbenchImage[] = [];
    const rejected: Array<{ name: string; error: string }> = [];
    for (let offset = 0; offset < files.length; offset += 200) {
      const form = new FormData();
      files.slice(offset, offset + 200).forEach((file) => form.append('files', file));
      const res = await fetch(`/api/workbench/sessions/${sessionId}/images`, { method: 'POST', body: form });
      if (!res.ok) await safeThrowError(res);
      const result = await res.json();
      images.push(...(result.images || []));
      rejected.push(...(result.rejected || []));
    }
    return { session_id: sessionId, images, rejected };
  },

  async deleteWorkbenchImages(sessionId: string, imageIds: string[]) {
    const res = await fetch(`/api/workbench/sessions/${sessionId}/images`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_ids: imageIds })
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async updateWorkbenchImageRoi(sessionId: string, imageId: string, roi: WorkbenchRoi | null) {
    const res = await fetch(`/api/workbench/sessions/${sessionId}/images/${imageId}/roi`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ roi })
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async rotateWorkbenchImage(sessionId: string, imageId: string, direction: 'clockwise' | 'counterclockwise') {
    const res = await fetch(`/api/workbench/sessions/${sessionId}/images/${imageId}/rotate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ direction })
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async inferWorkbench(sessionId: string, payload: any) {
    const res = await fetch(`/api/workbench/sessions/${sessionId}/infer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async inspectWorkbenchDataset(datasetPath: string) {
    const res = await fetch('/api/workbench/datasets/inspect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_path: datasetPath })
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async evaluateWorkbench(payload: any) {
    const res = await fetch('/api/workbench/evaluations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async listWorkbenchEvaluations(datasetPath: string) {
    const query = new URLSearchParams({ dataset_path: datasetPath });
    const res = await fetch(`/api/workbench/evaluations?${query.toString()}`);
    if (!res.ok) await safeThrowError(res);
    return res.json();
  },

  async getWorkbenchEvaluation(evaluationId: string, datasetPath: string) {
    const query = new URLSearchParams({ dataset_path: datasetPath });
    const res = await fetch(`/api/workbench/evaluations/${encodeURIComponent(evaluationId)}?${query.toString()}`);
    if (!res.ok) await safeThrowError(res);
    return res.json();
  }
};
