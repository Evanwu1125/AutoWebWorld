import { JSONPath } from 'jsonpath-plus';
import { cloneDeep, set, get } from 'lodash-es';

export class FSMRuntime {
  constructor(fsmData, context) {
    this.fsm = fsmData;
    this.context = context;
  }

  get currentPage() {
    return this.fsm.pages.find(p => p.id === this.context.currentPageId);
  }
}