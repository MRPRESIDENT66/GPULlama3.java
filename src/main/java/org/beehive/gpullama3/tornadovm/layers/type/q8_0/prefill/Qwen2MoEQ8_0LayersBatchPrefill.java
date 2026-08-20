package org.beehive.gpullama3.tornadovm.layers.type.q8_0.prefill;

import org.beehive.gpullama3.inference.state.Qwen2MoEState;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2MoETornadoWeights;
import org.beehive.gpullama3.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.Qwen2MoEBatchKernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerBatchPrefillKernels;
import org.beehive.gpullama3.tornadovm.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.List;
import java.util.stream.IntStream;

/** Batched-prefill Transformer-layer TaskGraphs for Qwen2-MoE Q8_0. */
public final class Qwen2MoEQ8_0LayersBatchPrefill
        implements BatchPrefillTransformerLayerTaskGraphs {

    private static final int LOCAL_WORK_GROUP_SIZE = 32;
    private static final int EXPERT_TILE_SIZE = 2;
    private static final int EXPERT_2D_LOCAL_WORK_GROUP_SIZE = 128;
    private static final int EXPERT_OUTPUT_ROWS_PER_GROUP =
            EXPERT_2D_LOCAL_WORK_GROUP_SIZE / LOCAL_WORK_GROUP_SIZE;

    private final Qwen2MoEState state;
    private final Qwen2MoETornadoWeights weights;
    private final Qwen2MoEConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final int dim;
    private final int kvDim;
    private final int topK;
    private final int numberOfAssignments;
    private final int sharedExpertTokenTile;
    private final List<ImmutableTaskGraph> layerTaskGraphs;
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
        this.dim = config.dim();
        this.kvDim = config.kvDim();
        this.topK = config.numberOfExpertsUsed();
        this.numberOfAssignments = batchSize * topK;
        this.sharedExpertTokenTile = batchSize >= 4 ? 4 : 2;
        this.layerTaskGraphs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::createBatchPrefillLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    /** Creates the complete batch-prefill TaskGraph for one Transformer layer. */
    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = "batchPrefillLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) {
            lastLayerTaskGraphID = graphName;
        }

        TaskGraph layer = new TaskGraph(graphName);
        configureDataTransfers(layer, layerIndex);
        configureAttention(layer, layerIndex);
        configureMoE(layer, layerIndex);
        layer.persistOnDevice(
                state.wrapXBatch,
                state.wrapKeyCache,
                state.wrapValueCache,
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
                weights.sharedGateInputLayered[layerIndex].asFloatArray(),
                state.wrapQuantizedFfnInputBatch,
                state.wrapQuantizedFfnInputScalesBatch);
        return layer;
    }

    /** Declares layer weights and the batch buffers that remain on the GPU. */
    private void configureDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    state.batchStartPosHolder,
                    state.activeBatchSizeHolder);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.attnScaleBatch,
                    state.ffnScaleBatch,
                    state.wrapXbBatch,
                    state.wrapQBatch,
                    state.wrapKBatch,
                    state.wrapVBatch,
                    state.wrapKeyCache,
                    state.wrapValueCache,
                    state.wrapRouterLogitsBatch,
                    state.wrapSelectedExpertsBatch,
                    state.wrapRoutingWeightsBatch,
                    state.wrapGroupedAssignmentIds,
                    state.wrapGroupedPositionByAssignment,
                    state.wrapExpertTileIds,
                    state.wrapExpertTileStarts,
                    state.wrapExpertTileCounts,
                    state.wrapExpertTileCountHolder,
                    state.wrapGroupedExpertHidden,
                    state.wrapGroupedExpertDown,
                    state.wrapSharedHiddenBatch,
                    state.wrapSharedWeightBatch,
                    state.wrapQuantizedFfnInputBatch,
                    state.wrapQuantizedFfnInputScalesBatch);
            layer.consumeFromDevice("prefillActivation", state.wrapXBatch);
        } else {
            String predecessor = "batchPrefillLayer_" + (layerIndex - 1);
            layer.consumeFromDevice(
                    predecessor,
                    context,
                    state.wrapXBatch,
                    state.wrapXbBatch,
                    state.wrapQBatch,
                    state.wrapKBatch,
                    state.wrapVBatch,
                    state.wrapKeyCache,
                    state.wrapValueCache,
                    state.batchStartPosHolder,
                    state.activeBatchSizeHolder,
                    state.attnScaleBatch,
                    state.ffnScaleBatch,
                    state.wrapRouterLogitsBatch,
                    state.wrapSelectedExpertsBatch,
                    state.wrapRoutingWeightsBatch,
                    state.wrapGroupedAssignmentIds,
                    state.wrapGroupedPositionByAssignment,
                    state.wrapExpertTileIds,
                    state.wrapExpertTileStarts,
                    state.wrapExpertTileCounts,
                    state.wrapExpertTileCountHolder,
                    state.wrapGroupedExpertHidden,
                    state.wrapGroupedExpertDown,
                    state.wrapSharedHiddenBatch,
                    state.wrapSharedWeightBatch,
                    state.wrapQuantizedFfnInputBatch,
                    state.wrapQuantizedFfnInputScalesBatch);
        }

        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
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

    /** Adds Qwen2 attention tasks for every token in the prefill batch. */
    private void configureAttention(TaskGraph layer, int layerIndex) {
        layer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.wrapXBatch,
                state.attnScaleBatch,
                dim,
                config.rmsNormEps(),
                LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                state.wrapXbBatch,
                state.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.attnScaleBatch,
                dim);

        layer.task(
                "batch_qkv",
                TransformerBatchPrefillKernels::batchedFusedQKVMatmulQ8,
                context,
                state.wrapXbBatch,
                state.wrapQBatch,
                state.wrapKBatch,
                state.wrapVBatch,
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                dim,
                kvDim,
                LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_qkv_bias",
                Qwen2MoEBatchKernels::batchedQKVBias,
                context,
                state.wrapQBatch,
                state.wrapKBatch,
                state.wrapVBatch,
                weights.q_biasLayered[layerIndex].asFloatArray(),
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray(),
                state.activeBatchSizeHolder,
                dim,
                kvDim);

        layer.task(
                "batch_rope_kv",
                Qwen2MoEBatchKernels::batchedRopeWithKVCacheQwen2,
                context,
                state.batchStartPosHolder,
                state.activeBatchSizeHolder,
                state.wrapQBatch,
                state.wrapKBatch,
                state.wrapVBatch,
                state.wrapKeyCache,
                state.wrapValueCache,
                kvDim,
                config.headSize(),
                layerIndex,
                config.contextLength(),
                dim,
                config.ropeTheta());

        layer.task(
                "batch_attention",
                TransformerBatchPrefillKernels::batchedFlashAttention,
                context,
                state.batchStartPosHolder,
                state.wrapQBatch,
                state.wrapKeyCache,
                state.wrapValueCache,
                state.wrapXbBatch,
                config.numberOfHeads(),
                config.headSize(),
                kvDim,
                config.kvMul(),
                layerIndex,
                config.contextLength(),
                dim);

        layer.task(
                "batch_attn_out",
                TransformerBatchPrefillKernels::batchedMatVecWithResidualQ8,
                context,
                state.wrapXbBatch,
                state.wrapXBatch,
                weights.woLayered[layerIndex].asByteArray(),
                dim,
                dim,
                LOCAL_WORK_GROUP_SIZE);
    }

    /** Adds batched routing, routed experts, and the shared expert. */
    private void configureMoE(TaskGraph layer, int layerIndex) {
        layer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.wrapXBatch,
                state.ffnScaleBatch,
                dim,
                config.rmsNormEps(),
                LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                state.wrapXbBatch,
                state.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.ffnScaleBatch,
                dim);

        layer.task(
                "batch_ffn_quantize_q8_0",
                Qwen2MoEBatchKernels::quantizeBatchFfnInputQ8_0,
                context,
                state.wrapXbBatch,
                state.wrapQuantizedFfnInputBatch,
                state.wrapQuantizedFfnInputScalesBatch,
                state.activeBatchSizeHolder,
                dim);

        layer.task(
                "batch_router",
                Qwen2MoEBatchKernels::batchedRouterProjection,
                context,
                state.wrapXbBatch,
                state.wrapRouterLogitsBatch,
                weights.routerGateLayered[layerIndex].asFloatArray(),
                state.activeBatchSizeHolder,
                dim,
                config.numberOfExperts(),
                LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_topk",
                Qwen2MoEBatchKernels::batchedSoftmaxAndTopK,
                context,
                state.wrapRouterLogitsBatch,
                state.wrapSelectedExpertsBatch,
                state.wrapRoutingWeightsBatch,
                state.activeBatchSizeHolder,
                config.numberOfExperts(),
                topK);

        layer.task(
                "batch_group_assignments",
                Qwen2MoEBatchKernels::groupAssignmentsAndBuildExpertTiles,
                context,
                state.wrapSelectedExpertsBatch,
                state.wrapGroupedAssignmentIds,
                state.wrapGroupedPositionByAssignment,
                state.wrapExpertTileIds,
                state.wrapExpertTileStarts,
                state.wrapExpertTileCounts,
                state.wrapExpertTileCountHolder,
                state.activeBatchSizeHolder,
                config.numberOfExperts(),
                topK,
                EXPERT_TILE_SIZE);

        layer.task(
                "batch_routed_gate_up",
                Qwen2MoEBatchKernels::tiled2DRoutedExpertsGateUpSwiGLUQ8_0Int8Activation,
                context,
                state.wrapQuantizedFfnInputBatch,
                state.wrapQuantizedFfnInputScalesBatch,
                state.wrapGroupedAssignmentIds,
                state.wrapExpertTileIds,
                state.wrapExpertTileStarts,
                state.wrapExpertTileCounts,
                state.wrapExpertTileCountHolder,
                weights.gateExpertsLayered[layerIndex].asByteArray(),
                weights.upExpertsLayered[layerIndex].asByteArray(),
                state.wrapGroupedExpertHidden,
                dim,
                config.moeHiddenDim(),
                topK,
                EXPERT_2D_LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_routed_down",
                Qwen2MoEBatchKernels::tiled2DRoutedExpertsDownQ8_0,
                context,
                state.wrapGroupedExpertHidden,
                state.wrapExpertTileIds,
                state.wrapExpertTileStarts,
                state.wrapExpertTileCounts,
                state.wrapExpertTileCountHolder,
                weights.downExpertsLayered[layerIndex].asByteArray(),
                state.wrapGroupedExpertDown,
                dim,
                config.moeHiddenDim(),
                EXPERT_2D_LOCAL_WORK_GROUP_SIZE);

        layer.task(
                "batch_routed_accumulate",
                Qwen2MoEBatchKernels::accumulateGroupedRoutedExperts,
                context,
                state.wrapGroupedExpertDown,
                state.wrapGroupedPositionByAssignment,
                state.wrapRoutingWeightsBatch,
                state.wrapXBatch,
                state.activeBatchSizeHolder,
                dim,
                topK);

        configureSharedExpert(layer, layerIndex);
    }

    /** Adds the tiled shared-expert Gate/Up, gate-weight, and Down tasks. */
    private void configureSharedExpert(TaskGraph layer, int layerIndex) {
        if (sharedExpertTokenTile == 4) {
            layer.task(
                    "batch_shared_gate_up",
                    Qwen2MoEBatchKernels::tiled4SharedExpertGateUpSwiGLUQ8_0,
                    context,
                    state.wrapXbBatch,
                    state.activeBatchSizeHolder,
                    weights.sharedGateLayered[layerIndex].asByteArray(),
                    weights.sharedUpLayered[layerIndex].asByteArray(),
                    state.wrapSharedHiddenBatch,
                    dim,
                    config.sharedExpertHiddenDim(),
                    EXPERT_2D_LOCAL_WORK_GROUP_SIZE);
        } else {
            layer.task(
                    "batch_shared_gate_up",
                    Qwen2MoEBatchKernels::tiled2SharedExpertGateUpSwiGLUQ8_0,
                    context,
                    state.wrapXbBatch,
                    state.activeBatchSizeHolder,
                    weights.sharedGateLayered[layerIndex].asByteArray(),
                    weights.sharedUpLayered[layerIndex].asByteArray(),
                    state.wrapSharedHiddenBatch,
                    dim,
                    config.sharedExpertHiddenDim(),
                    EXPERT_2D_LOCAL_WORK_GROUP_SIZE);
        }

        layer.task(
                "batch_shared_weight",
                Qwen2MoEBatchKernels::batchedSharedExpertGateWeight,
                context,
                state.wrapXbBatch,
                weights.sharedGateInputLayered[layerIndex].asFloatArray(),
                state.wrapSharedWeightBatch,
                state.activeBatchSizeHolder,
                dim,
                LOCAL_WORK_GROUP_SIZE);

        if (sharedExpertTokenTile == 4) {
            layer.task(
                    "batch_shared_down",
                    Qwen2MoEBatchKernels::tiled4SharedExpertDownAndAccumulateQ8_0,
                    context,
                    state.wrapSharedHiddenBatch,
                    state.wrapSharedWeightBatch,
                    state.activeBatchSizeHolder,
                    weights.sharedDownLayered[layerIndex].asByteArray(),
                    state.wrapXBatch,
                    dim,
                    config.sharedExpertHiddenDim(),
                    EXPERT_2D_LOCAL_WORK_GROUP_SIZE);
        } else {
            layer.task(
                    "batch_shared_down",
                    Qwen2MoEBatchKernels::tiled2SharedExpertDownAndAccumulateQ8_0,
                    context,
                    state.wrapSharedHiddenBatch,
                    state.wrapSharedWeightBatch,
                    state.activeBatchSizeHolder,
                    weights.sharedDownLayered[layerIndex].asByteArray(),
                    state.wrapXBatch,
                    dim,
                    config.sharedExpertHiddenDim(),
                    EXPERT_2D_LOCAL_WORK_GROUP_SIZE);
        }
    }

    /** Configures the fixed WorkerGrid used by every task in every layer. */
    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid rmsWorker = groupedRowsWorker(batchSize);
        WorkerGrid batchScalarWorker = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid batchElementWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid qkvWorker = groupedRowsWorker(batchSize * (dim + 2 * kvDim));
        int qkvBiasGlobalWork = batchSize * (dim + 2 * kvDim);
        WorkerGrid qkvBiasWorker =
                WorkerGridFactory.genericWorker(
                        qkvBiasGlobalWork,
                        validLocalSize(qkvBiasGlobalWork, 256));

        int ropeGlobalWork = batchSize * (dim / 2);
        int ropeLocalWork = validLocalSize(ropeGlobalWork, 512);
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobalWork, ropeLocalWork);

        int attentionLocalWork = validLocalSize(config.headSize(), 64);
        WorkerGrid attentionWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * config.numberOfHeads() * attentionLocalWork,
                        attentionLocalWork);

        WorkerGrid batchDimRowsWorker = groupedRowsWorker(batchSize * dim);
        WorkerGrid routerWorker = groupedRowsWorker(batchSize * config.numberOfExperts());
        WorkerGrid groupingWorker = WorkerGridFactory.createSingleWorker();
        WorkerGrid routedHiddenWorker = int8ActivationRoutedHiddenWorker();
        WorkerGrid routedDownWorker = tiledExpertRowsWorker(dim);
        WorkerGrid tiledSharedHiddenWorker =
                sharedExpertTiledRowsWorker(
                        config.sharedExpertHiddenDim(), sharedExpertTokenTile);
        WorkerGrid tiledSharedDownWorker =
                sharedExpertTiledRowsWorker(dim, sharedExpertTokenTile);
        WorkerGrid sharedWeightWorker = groupedRowsWorker(batchSize);

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            String prefix = "batchPrefillLayer_" + layer + ".";
            scheduler.addWorkerGrid(prefix + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(prefix + "batch_attn_rms_apply", batchElementWorker);
            scheduler.addWorkerGrid(prefix + "batch_qkv", qkvWorker);
            scheduler.addWorkerGrid(prefix + "batch_qkv_bias", qkvBiasWorker);
            scheduler.addWorkerGrid(prefix + "batch_rope_kv", ropeWorker);
            scheduler.addWorkerGrid(prefix + "batch_attention", attentionWorker);
            scheduler.addWorkerGrid(prefix + "batch_attn_out", batchDimRowsWorker);
            scheduler.addWorkerGrid(prefix + "batch_ffn_rms", rmsWorker);
            scheduler.addWorkerGrid(prefix + "batch_ffn_rms_apply", batchElementWorker);
            scheduler.addWorkerGrid(prefix + "batch_ffn_quantize_q8_0", ffnQuantizeWorker());
            scheduler.addWorkerGrid(prefix + "batch_router", routerWorker);
            scheduler.addWorkerGrid(prefix + "batch_topk", batchScalarWorker);
            scheduler.addWorkerGrid(prefix + "batch_group_assignments", groupingWorker);
            scheduler.addWorkerGrid(prefix + "batch_routed_gate_up", routedHiddenWorker);
            scheduler.addWorkerGrid(prefix + "batch_routed_down", routedDownWorker);
            scheduler.addWorkerGrid(prefix + "batch_routed_accumulate", batchElementWorker);
            scheduler.addWorkerGrid(prefix + "batch_shared_gate_up", tiledSharedHiddenWorker);
            scheduler.addWorkerGrid(prefix + "batch_shared_weight", sharedWeightWorker);
            scheduler.addWorkerGrid(prefix + "batch_shared_down", tiledSharedDownWorker);
        }
    }

    /** Creates a 32-thread work-group for every logical output row. */
    private static WorkerGrid groupedRowsWorker(int rows) {
        return WorkerGridFactory.genericWorker(rows * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
    }

    /** Creates one 128-thread work-group for four output rows in every expert tile. */
    private WorkerGrid tiledExpertRowsWorker(int outputRows) {
        int rowTiles =
                (outputRows + EXPERT_OUTPUT_ROWS_PER_GROUP - 1) / EXPERT_OUTPUT_ROWS_PER_GROUP;
        int workGroups = numberOfAssignments * rowTiles;
        return WorkerGridFactory.genericWorker(
                workGroups * EXPERT_2D_LOCAL_WORK_GROUP_SIZE, EXPERT_2D_LOCAL_WORK_GROUP_SIZE);
    }

    /** Uses eight threads per output row because each thread executes one DP4A. */
    private WorkerGrid int8ActivationRoutedHiddenWorker() {
        int rowsPerGroup = LOCAL_WORK_GROUP_SIZE / 8;
        int rowTiles = (config.moeHiddenDim() + rowsPerGroup - 1) / rowsPerGroup;
        int workGroups = numberOfAssignments * rowTiles;
        return WorkerGridFactory.genericWorker(workGroups * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
    }

    /** Uses one 32-thread work-group to quantize each 32-value activation block. */
    private WorkerGrid ffnQuantizeWorker() {
        int blocksPerToken = (dim + LOCAL_WORK_GROUP_SIZE - 1) / LOCAL_WORK_GROUP_SIZE;
        int workGroups = batchSize * blocksPerToken;
        return WorkerGridFactory.genericWorker(workGroups * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
    }

    /** Creates a token-tiled, four-output-row shared-expert grid. */
    private WorkerGrid sharedExpertTiledRowsWorker(int outputRows, int tokensPerTile) {
        int tokenTiles = (batchSize + tokensPerTile - 1) / tokensPerTile;
        int rowTiles =
                (outputRows + EXPERT_OUTPUT_ROWS_PER_GROUP - 1)
                        / EXPERT_OUTPUT_ROWS_PER_GROUP;
        int workGroups = tokenTiles * rowTiles;
        return WorkerGridFactory.genericWorker(
                workGroups * EXPERT_2D_LOCAL_WORK_GROUP_SIZE,
                EXPERT_2D_LOCAL_WORK_GROUP_SIZE);
    }

    /** Finds a legal local size that divides the requested global dimension. */
    private static int validLocalSize(int size, int maximum) {
        int localSize = Math.min(size, maximum);
        while (localSize > 1 && size % localSize != 0) {
            localSize--;
        }
        return localSize;
    }

    @Override
    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return layerTaskGraphs;
    }

    @Override
    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    public KernelContext getContext() {
        return context;
    }
}
