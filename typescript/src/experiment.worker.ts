import { runExperiment } from './experiment.ts';
import type { ExperimentOptions } from './experiment.ts';

self.onmessage = (event: MessageEvent<{ id: number; options: ExperimentOptions }>) => {
  const { id, options } = event.data;
  try {
    self.postMessage({ type: 'result', id, result: runExperiment(options) });
  } catch (error) {
    self.postMessage({ type: 'error', id, message: error instanceof Error ? error.message : 'The comparison could not be calculated.' });
  }
};
