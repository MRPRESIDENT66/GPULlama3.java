package org.beehive.gpullama3.tornadovm.layers;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.List;

/**
 * Interface for a group of N batched-prefill transformer-layer TornadoVM TaskGraphs.
 *
 * <p>Implemented by {@code LlamaFP16LayersBatchPrefillMMA}, {@code LlamaFP16LayersBatchPrefill}, {@code LlamaQ8_0LayersBatchPrefillMMA} and {@code LlamaQ8_0LayersBatchPrefill}.</p>
 */
public interface BatchPrefillTransformerLayerTaskGraphs {
    List<ImmutableTaskGraph> getLayerImmutableTaskGraphs();

    void updateGridScheduler(GridScheduler scheduler);

    /**
     * Updates batch-layer WorkerGrids for a runtime batch size.
     * Implementations that do not support dynamic sizing keep their normal grids.
     */
    default void updateGridScheduler(GridScheduler scheduler, int runtimeBatchSize) {
        updateGridScheduler(scheduler);
    }

    String getLastLayerTaskGraphID();
}
