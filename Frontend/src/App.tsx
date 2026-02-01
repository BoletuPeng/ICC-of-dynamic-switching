import { ReactFlowProvider } from '@xyflow/react';
import { Header } from './components/Header';
import { FlowCanvas } from './components/Flow';
import { Sidebar } from './components/Sidebar';
import { NodeEditor } from './components/NodeEditor';

// =============================================================================
// App Component
// =============================================================================

export default function App() {
  return (
    <ReactFlowProvider>
      <div className="app">
        <Header />
        <div className="app-content">
          <FlowCanvas />
          <Sidebar />
        </div>
        <NodeEditor />
      </div>
    </ReactFlowProvider>
  );
}
