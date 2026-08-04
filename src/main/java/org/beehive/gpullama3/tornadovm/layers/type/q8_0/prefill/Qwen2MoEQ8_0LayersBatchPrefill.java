package org.beehive.gpullama3.tornadovm.layers.type.q8_0.prefill;

import org.beehive.gpullama3.inference.state.Qwen2MoEState;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2MoETornadoWeights;
import org.beehive.gpullama3.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.Qwen2MoEBatchKernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerBatchPrefillKernels;
import org.beehive.gpullama3.tornadovm.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;
import org.beehive.gpullama3.validation.MoECorrectnessTrace;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Fixed-graph, expert-grouped Q8_0 batch-prefill layers for Qwen2-MoE.
 *
 * <p>The graph shape and maximum WorkerGrids are static. Router results are
 * converted on the GPU into expert counts, offsets, and a grouped assignment
 * permutation that changes for every batch.</p>
 */
public final class Qwen2MoEQ8_0LayersBatchPrefill
        implements BatchPrefillTransformerLayerTaskGraphs {

    private static final int LOCAL_WORK_GROUP_SIZE = 32;

    private final Qwen2MoEState state;
    private final Qwen2MoETornadoWeights weights;
    private final Qwen2MoEConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public Qwen2MoEQ8_0LayersBatchPrefill(
            Qwen2MoEState state,
            Qwen2MoETornadoWeights weights,
            Qwen2MoEConfiguration config,
            int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.layerITGs = IntStream.range(0, config.numberOfLayers())
                .mapToObj(this::createBatchPrefillLayerTaskGraph)
                .map(TaskGraph::snapshot)
                .toList();
    }

    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = "batchPrefillLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) {
            lastLayerTaskGraphID = graphName;
        }

        TaskGraph layer = new TaskGraph(graphName);
        configureDataFlow(layer, layerIndex);
        transferLayerWeights(layer, layerIndex);

        int dim = config.dim();
        int kvDim = config.kvDim();
        int experts = config.numberOfExperts();
        int topK = config.numberOfExpertsUsed();
        int moeHiddenDim = config.moeHiddenDim();
        int sharedHiddenDim = config.sharedExpertHiddenDim();

        // Qwen2 attention for all tokens in the prefill chunk.
        layer.task("batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context, state.wrapXBatch, state.attnScaleBatch,
                dim, config.rmsNormEps(), LOCAL_WORK_GROUP_SIZE);
        layer.task("batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context, state.wrapXbBatch, state.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.attnScaleBatch, dim);
        layer.task("batch_qkv",
                TransformerBatchPrefillKernels::batchedFusedQKVMatmulQ8,
                context, state.wrapXbBatch,
                state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                dim, kvDim, LOCAL_WORK_GROUP_SIZE);
        layer.task("batch_qkv_bias",
                Qwen2MoEBatchKernels::batchedQKVBias,
                context, state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                weights.q_biasLayered[layerIndex].asFloatArray(),
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray(),
                state.activeBatchSizeHolder, dim, kvDim);
        layer.task("batch_rope_kv",
                Qwen2MoEBatchKernels::batchedQwen2RoPEWithKVCache,
                context, state.batchStartPosHolder,
                state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                state.wrapKeyCache, state.wrapValueCache,
                state.activeBatchSizeHolder, config.numberOfKeyValueHeads(), config.headSize(), kvDim,
                layerIndex, config.contextLength(), dim, config.ropeTheta());
        layer.task("batch_attention",
                TransformerBatchPrefillKernels::batchedFlashAttention,
                context, state.batchStartPosHolder,
                state.wrapQBatch, state.wrapKeyCache, state.wrapValueCache,
                state.wrapXbBatch,
                config.numberOfHeads(), config.headSize(), kvDim, config.kvMul(),
                layerIndex, config.contextLength(), dim);
        layer.task("batch_attn_out",
                TransformerBatchPrefillKernels::batchedMatVecWithResidualQ8,
                context, state.wrapXbBatch, state.wrapXBatch,
                weights.woLayered[layerIndex].asByteArray(),
                dim, dim, LOCAL_WORK_GROUP_SIZE);

        // Normalize once, then route all tokens.
        layer.task("batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedFFNRmsReduceParallel,
                context, state.wrapXBatch, state.ffnScaleBatch,
                dim, config.rmsNormEps(), LOCAL_WORK_GROUP_SIZE);
        layer.task("batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context, state.wrapXbBatch, state.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.ffnScaleBatch, dim);
        layer.task("batch_router_projection",
                Qwen2MoEBatchKernels::batchedRouterProjection,
                context, state.wrapXbBatch, state.wrapRouterLogitsBatch,
                weights.routerGateLayered[layerIndex].asFloatArray(),
                state.activeBatchSizeHolder, dim, experts, LOCAL_WORK_GROUP_SIZE);
        if (MoECorrectnessTrace.isEnabled()) {
            layer.task("batch_router_trace_copy",
                    Qwen2MoEBatchKernels::copyBatchedRouterLogits,
                    context, state.wrapRouterLogitsBatch, state.wrapRawRouterLogitsBatch,
                    state.activeBatchSizeHolder, experts);
        }
        layer.task("batch_router_topk",
                Qwen2MoEBatchKernels::batchedSoftmaxAndTopK,
                context, state.wrapRouterLogitsBatch,
                state.wrapSelectedExpertsBatch, state.wrapRoutingWeightsBatch,
                state.activeBatchSizeHolder, experts, topK);

        // Build counts/offsets/permutation on GPU without changing the TaskGraph.
        layer.task("batch_group_experts",
                Qwen2MoEBatchKernels::groupAssignmentsByExpert,
                context, state.wrapSelectedExpertsBatch,
                state.wrapExpertCounts, state.wrapExpertOffsets,
                state.wrapGroupedAssignmentIds, state.wrapGroupedPositionByAssignment,
                state.activeBatchSizeHolder, experts, topK);

        // Process routed assignments in expert-major order, then scatter to tokens.
        layer.task("batch_grouped_gate_up",
                Qwen2MoEBatchKernels::groupedGateUpSwiGLUQ8_0,
                context, state.wrapXbBatch,
                state.wrapSelectedExpertsBatch, state.wrapGroupedAssignmentIds,
                weights.gateExpertsLayered[layerIndex].asByteArray(),
                weights.upExpertsLayered[layerIndex].asByteArray(),
                state.wrapGroupedExpertHidden,
                state.activeBatchSizeHolder, dim, moeHiddenDim, experts, topK, LOCAL_WORK_GROUP_SIZE);
        layer.task("batch_grouped_down",
                Qwen2MoEBatchKernels::groupedDownQ8_0,
                context, state.wrapGroupedExpertHidden, state.wrapGroupedExpertDown,
                state.wrapSelectedExpertsBatch, state.wrapGroupedAssignmentIds,
                state.wrapRoutingWeightsBatch,
                weights.downExpertsLayered[layerIndex].asByteArray(),
                state.activeBatchSizeHolder, dim, moeHiddenDim, experts, topK, LOCAL_WORK_GROUP_SIZE);
        layer.task("batch_scatter_routed",
                Qwen2MoEBatchKernels::scatterGroupedRoutedOutput,
                context, state.wrapGroupedExpertDown, state.wrapXBatch,
                state.wrapGroupedPositionByAssignment,
                state.activeBatchSizeHolder, dim, topK);

        // Shared expert remains dense across the batch and always executes.
        layer.task("batch_shared_gate_up",
                Qwen2MoEBatchKernels::batchedSharedGateUpSwiGLUQ8_0,
                context, state.wrapXbBatch,
                weights.sharedGateLayered[layerIndex].asByteArray(),
                weights.sharedUpLayered[layerIndex].asByteArray(),
                state.wrapSharedHiddenBatch,
                weights.sharedGateInputLayered[layerIndex].asFloatArray(),
                state.wrapSharedWeightBatch,
                state.activeBatchSizeHolder, dim, sharedHiddenDim, LOCAL_WORK_GROUP_SIZE);
        layer.task("batch_shared_down",
                Qwen2MoEBatchKernels::batchedSharedDownAndAccumulateQ8_0,
                context, state.wrapSharedHiddenBatch,
                weights.sharedDownLayered[layerIndex].asByteArray(),
                state.wrapSharedWeightBatch, state.wrapXBatch,
                state.activeBatchSizeHolder, dim, sharedHiddenDim, LOCAL_WORK_GROUP_SIZE);

        // The decode graph consumes these exact device buffers, avoiding a second
        // resident copy of the large MoE weights.
        layer.persistOnDevice(
                state.wrapXBatch, state.wrapKeyCache, state.wrapValueCache,
                state.activeBatchSizeHolder,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.q_biasLayered[layerIndex].asFloatArray(),
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.routerGateLayered[layerIndex].asFloatArray(),
                weights.gateExpertsLayered[layerIndex].asByteArray(),
                weights.upExpertsLayered[layerIndex].asByteArray(),
                weights.downExpertsLayered[layerIndex].asByteArray(),
                weights.sharedGateLayered[layerIndex].asByteArray(),
                weights.sharedUpLayered[layerIndex].asByteArray(),
                weights.sharedDownLayered[layerIndex].asByteArray(),
                weights.sharedGateInputLayered[layerIndex].asFloatArray());
        if (MoECorrectnessTrace.isEnabled()) {
            layer.persistOnDevice(state.wrapRawRouterLogitsBatch);
            layer.transferToHost(DataTransferMode.EVERY_EXECUTION,
                    state.wrapRawRouterLogitsBatch, state.wrapSelectedExpertsBatch,
                    state.wrapRoutingWeightsBatch);
        }
        return layer;
    }

    private void configureDataFlow(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    state.batchStartPosHolder, state.activeBatchSizeHolder);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.attnScaleBatch, state.ffnScaleBatch,
                    state.wrapXbBatch, state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.wrapRouterLogitsBatch, state.wrapSelectedExpertsBatch,
                    state.wrapRoutingWeightsBatch, state.wrapExpertCounts,
                    state.wrapExpertOffsets, state.wrapGroupedAssignmentIds,
                    state.wrapGroupedPositionByAssignment,
                    state.wrapGroupedExpertHidden, state.wrapGroupedExpertDown,
                    state.wrapSharedHiddenBatch, state.wrapSharedWeightBatch);
            if (MoECorrectnessTrace.isEnabled()) {
                layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        state.wrapRawRouterLogitsBatch);
            }
            layer.consumeFromDevice("prefillActivation", state.wrapXBatch);
        } else {
            String predecessor = "batchPrefillLayer_" + (layerIndex - 1);
            layer.consumeFromDevice(predecessor,
                    context,
                    state.wrapXBatch, state.wrapXbBatch,
                    state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.batchStartPosHolder, state.activeBatchSizeHolder,
                    state.attnScaleBatch, state.ffnScaleBatch,
                    state.wrapRouterLogitsBatch, state.wrapSelectedExpertsBatch,
                    state.wrapRoutingWeightsBatch, state.wrapExpertCounts,
                    state.wrapExpertOffsets, state.wrapGroupedAssignmentIds,
                    state.wrapGroupedPositionByAssignment,
                    state.wrapGroupedExpertHidden, state.wrapGroupedExpertDown,
                    state.wrapSharedHiddenBatch, state.wrapSharedWeightBatch);
            if (MoECorrectnessTrace.isEnabled()) {
                layer.consumeFromDevice(predecessor, state.wrapRawRouterLogitsBatch);
            }
        }
    }

    private void transferLayerWeights(TaskGraph layer, int layerIndex) {
        layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.q_biasLayered[layerIndex].asFloatArray(),
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.routerGateLayered[layerIndex].asFloatArray(),
                weights.gateExpertsLayered[layerIndex].asByteArray(),
                weights.upExpertsLayered[layerIndex].asByteArray(),
                weights.downExpertsLayered[layerIndex].asByteArray(),
                weights.sharedGateLayered[layerIndex].asByteArray(),
                weights.sharedUpLayered[layerIndex].asByteArray(),
                weights.sharedDownLayered[layerIndex].asByteArray(),
                weights.sharedGateInputLayered[layerIndex].asFloatArray());
    }

    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        updateGridScheduler(scheduler, batchSize);
    }

    /** Updates the same TaskGraphs with either the prefill or decode batch size. */
    @Override
    public void updateGridScheduler(GridScheduler scheduler, int runtimeBatchSize) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int experts = config.numberOfExperts();
        int assignments = runtimeBatchSize * config.numberOfExpertsUsed();

        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(
                runtimeBatchSize * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
        WorkerGrid elementWorker = WorkerGridFactory.genericWorker(runtimeBatchSize * dim, 256);
        WorkerGrid qkvWorker = rowWorker(runtimeBatchSize * (dim + 2 * kvDim));
        WorkerGrid qkvBiasWorker = WorkerGridFactory.genericWorker(
                runtimeBatchSize * (dim + 2 * kvDim), 256);
        int ropeGlobal = runtimeBatchSize * (dim / 2);
        int ropeLocal = divisorAtMost(ropeGlobal, 512);
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);
        int attentionLocal = divisorAtMost(config.headSize(), 64);
        WorkerGrid attentionWorker = WorkerGridFactory.genericWorker(
                runtimeBatchSize * config.numberOfHeads() * attentionLocal, attentionLocal);
        WorkerGrid dimRowWorker = rowWorker(runtimeBatchSize * dim);
        WorkerGrid routerWorker = rowWorker(runtimeBatchSize * experts);
        WorkerGrid routerTraceWorker = WorkerGridFactory.genericWorker(
                runtimeBatchSize * experts,
                divisorAtMost(runtimeBatchSize * experts, LOCAL_WORK_GROUP_SIZE));
        WorkerGrid topKWorker = WorkerGridFactory.genericWorker(runtimeBatchSize, 1);
        WorkerGrid groupingWorker = WorkerGridFactory.genericWorker(1, 1);
        WorkerGrid groupedGateUpWorker = rowWorker(assignments * config.moeHiddenDim());
        WorkerGrid groupedDownWorker = rowWorker(assignments * dim);
        WorkerGrid sharedGateUpWorker = rowWorker(runtimeBatchSize * config.sharedExpertHiddenDim());

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            String prefix = "batchPrefillLayer_" + layer + ".";
            scheduler.addWorkerGrid(prefix + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(prefix + "batch_attn_rms_apply", elementWorker);
            scheduler.addWorkerGrid(prefix + "batch_qkv", qkvWorker);
            scheduler.addWorkerGrid(prefix + "batch_qkv_bias", qkvBiasWorker);
            scheduler.addWorkerGrid(prefix + "batch_rope_kv", ropeWorker);
            scheduler.addWorkerGrid(prefix + "batch_attention", attentionWorker);
            scheduler.addWorkerGrid(prefix + "batch_attn_out", dimRowWorker);
            scheduler.addWorkerGrid(prefix + "batch_ffn_rms", rmsWorker);
            scheduler.addWorkerGrid(prefix + "batch_ffn_rms_apply", elementWorker);
            scheduler.addWorkerGrid(prefix + "batch_router_projection", routerWorker);
            if (MoECorrectnessTrace.isEnabled()) {
                scheduler.addWorkerGrid(prefix + "batch_router_trace_copy", routerTraceWorker);
            }
            scheduler.addWorkerGrid(prefix + "batch_router_topk", topKWorker);
            scheduler.addWorkerGrid(prefix + "batch_group_experts", groupingWorker);
            scheduler.addWorkerGrid(prefix + "batch_grouped_gate_up", groupedGateUpWorker);
            scheduler.addWorkerGrid(prefix + "batch_grouped_down", groupedDownWorker);
            scheduler.addWorkerGrid(prefix + "batch_scatter_routed", elementWorker);
            scheduler.addWorkerGrid(prefix + "batch_shared_gate_up", sharedGateUpWorker);
            scheduler.addWorkerGrid(prefix + "batch_shared_down", dimRowWorker);
        }
    }

    private static WorkerGrid rowWorker(int rows) {
        return WorkerGridFactory.genericWorker(rows * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
    }

    private static int divisorAtMost(int value, int maximum) {
        int candidate = Math.min(value, maximum);
        while (candidate > 1 && value % candidate != 0) {
            candidate--;
        }
        return candidate;
    }

    @Override
    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return layerITGs;
    }

    @Override
    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    public KernelContext getContext() {
        return context;
    }
}
