package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerBatchPrefillKernels;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Lightweight placeholders required by the existing 2N+3 graph layout.
 * Qwen2-MoE decode computation has already run through the fixed batch layers;
 * these graphs only relay row zero and the KV cache to the logits graph.
 */
public final class MoEDecodeRelayLayerTaskGraphs implements TransformerLayerTaskGraphs {

    private final List<ImmutableTaskGraph> graphs;
    private final String lastGraphId;

    public MoEDecodeRelayLayerTaskGraphs(State state, Configuration config) {
        this.graphs = IntStream.range(0, config.numberOfLayers())
                .mapToObj(layer -> createRelayGraph(state, layer))
                .toList();
        this.lastGraphId = relayGraphName(config.numberOfLayers() - 1);
    }

    private static ImmutableTaskGraph createRelayGraph(State state, int layer) {
        String graphName = relayGraphName(layer);
        String predecessor = layer == 0 ? "decodeActivation" : relayGraphName(layer - 1);
        KernelContext context = new KernelContext();
        return new TaskGraph(graphName)
                .consumeFromDevice(predecessor,
                        state.wrapX, state.wrapKeyCache, state.wrapValueCache)
                .task("relay", TransformerBatchPrefillKernels::batchPassthrough,
                        context, state.wrapX)
                .persistOnDevice(state.wrapX, state.wrapKeyCache, state.wrapValueCache)
                .snapshot();
    }

    private static String relayGraphName(int layer) {
        return "moeDecodeRelay_" + layer;
    }

    @Override
    public List<ImmutableTaskGraph> getFFNLayerImmutableTaskGraphs() {
        return graphs;
    }

    @Override
    public String getLastFFNLayerTaskGraphID() {
        return lastGraphId;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        for (int layer = 0; layer < graphs.size(); layer++) {
            scheduler.addWorkerGrid(relayGraphName(layer) + ".relay",
                    WorkerGridFactory.genericWorker(1, 1));
        }
        return scheduler;
    }
}
