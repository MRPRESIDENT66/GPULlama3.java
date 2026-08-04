package org.beehive.gpullama3.tornadovm.plan.components.activation;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.state.Qwen2MoEState;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.kernels.Qwen2MoEBatchKernels;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Bridges the final active row of a batch graph to the single-token logits graph. */
public final class BatchFirstTokenDecodeActivation implements ActivationTaskGraph {

    private final ImmutableTaskGraph graph;
    private final int dim;
    private final IntArray activeBatchSizeHolder;

    public BatchFirstTokenDecodeActivation(
            State state,
            Configuration config,
            String lastBatchLayerId) {
        this.dim = config.dim();
        this.activeBatchSizeHolder = ((Qwen2MoEState) state).activeBatchSizeHolder;
        KernelContext context = new KernelContext();
        this.graph = new TaskGraph("decodeActivation")
                .consumeFromDevice(lastBatchLayerId,
                        state.wrapXBatch, state.wrapKeyCache, state.wrapValueCache,
                        activeBatchSizeHolder)
                .task("copy_last_active_batch_token",
                        Qwen2MoEBatchKernels::copyLastActiveTokenActivation,
                        context, state.wrapXBatch, state.wrapX,
                        activeBatchSizeHolder, dim)
                .persistOnDevice(state.wrapX, state.wrapKeyCache, state.wrapValueCache,
                        activeBatchSizeHolder)
                .snapshot();
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return graph;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        scheduler.addWorkerGrid("decodeActivation.copy_last_active_batch_token",
                WorkerGridFactory.genericWorker(dim, 128));
        return scheduler;
    }
}
