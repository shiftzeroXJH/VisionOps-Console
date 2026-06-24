export type MetricGoal = {
  metric: string;
  target: number;
};

export type Experiment = {
  experiment_id: string;
  description: string;
  project: string;
  status: string;
  task_type: string;
  dataset_root: string;
  dataset_yaml?: string;
  pretrained_model: string;
  goal: MetricGoal;
  trial_count: number;
  best_metric?: {
    trial_id: string;
    iteration: number;
    metric: string;
    value: number;
  };
  latest_trial?: any; // Define properly if needed
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

  async updateSettings(payload: { yolo_python: string }) {
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
  }
};
