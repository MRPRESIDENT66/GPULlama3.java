package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.Qwen2MoEState;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2MoETornadoWeights;
import org.beehive.gpullama3.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.Qwen2Kernels;
import org.beehive.gpullama3.tornadovm.kernels.Qwen2MoEKernels;
import org.beehive.gpullama3.tornadovm.kernels.Qwen3Kernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;
import org.beehive.gpullama3.validation.MoECorrectnessTrace;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Single-token Q8_0 TaskGraphs for Qwen2-MoE / Qwen1.5-MoE.
 *
 * <p>The attention block follows Qwen2. Its dense FFN is replaced by the
 * routed-expert pipeline: normalize, route, choose top-K experts, execute each
 * selected expert, and accumulate its weighted output into {@code wrapX}.</p>
 */
public final class Qwen2MoEQ8_0FFNLayers
        extends AbstractTransformerLayerTaskGraphs<Qwen2MoETornadoWeights, Qwen2MoEConfiguration> {

    private final Qwen2MoEState moeState;

    public Qwen2MoEQ8_0FFNLayers(String taskGraphName,
                                  Qwen2MoEState state,
                                  Qwen2MoETornadoWeights weights,
                                  Qwen2MoEConfiguration config,
                                  SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.moeState = state;
        setupFFNLayers();
    }

    /** Sets the GPU worker grid for each task in each Transformer layer. */
    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), moeState.localSize);

        WorkerGrid qkvWorker = workerForRows(config.dim() + 2 * config.kvDim());
        WorkerGrid qkvBiasWorker = new WorkerGrid1D(config.dim());
        qkvBiasWorker.setGlobalWork(config.dim(), 1, 1);
        qkvBiasWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        WorkerGrid ropeWorker = new WorkerGrid2D(config.numberOfHeads(), config.headSize() / 2);
        ropeWorker.setGlobalWork(config.numberOfHeads(), config.headSize() / 2, 1);
        ropeWorker.setLocalWork(1, 1, 1);

        int attentionLocalSize = Math.min(config.headSize(), 64);
        WorkerGrid attentionWorker = new WorkerGrid1D(config.numberOfHeads() * attentionLocalSize);
        attentionWorker.setLocalWork(attentionLocalSize, 1, 1);

        WorkerGrid dimElementWorker = new WorkerGrid1D(config.dim());
        dimElementWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);
        WorkerGrid dimWorker = workerForRows(config.dim());
        WorkerGrid routerWorker = workerForRows(config.numberOfExperts());
        int routerCopyGlobalSize = ((config.numberOfExperts() + LOCAL_WORK_GROUP_SIZE_ALLOC - 1)
                / LOCAL_WORK_GROUP_SIZE_ALLOC) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid routerCopyWorker = new WorkerGrid1D(routerCopyGlobalSize);
        routerCopyWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);
        WorkerGrid topKWorker = new WorkerGrid1D(LOCAL_WORK_GROUP_SIZE_ALLOC);
        topKWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);
        WorkerGrid expertHiddenWorker = workerForRows(config.moeHiddenDim());
        WorkerGrid sharedHiddenWorker = workerForRows(config.sharedExpertHiddenDim());

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            String prefix = "layer_" + layer + ".";
            scheduler.addWorkerGrid(prefix + "attn_rms_reduce", rmsNormWorker);
            scheduler.addWorkerGrid(prefix + "attn_rms_qkv_projection", qkvWorker);
            scheduler.addWorkerGrid(prefix + "fused_qkv_bias", qkvBiasWorker);
            scheduler.addWorkerGrid(prefix + "rope_and_kv_cache", ropeWorker);
            scheduler.addWorkerGrid(prefix + "attention", attentionWorker);
            scheduler.addWorkerGrid(prefix + "attn_output_proj", dimWorker);
            scheduler.addWorkerGrid(prefix + "ffn_rms_reduce", rmsNormWorker);
            scheduler.addWorkerGrid(prefix + "ffn_rms_apply", dimElementWorker);
            scheduler.addWorkerGrid(prefix + "router_projection", routerWorker);
            if (MoECorrectnessTrace.isEnabled()) {
                scheduler.addWorkerGrid(prefix + "router_trace_copy", routerCopyWorker);
            }
            scheduler.addWorkerGrid(prefix + "router_softmax_topk", topKWorker);
            for (int slot = 0; slot < config.numberOfExpertsUsed(); slot++) {
                scheduler.addWorkerGrid(prefix + "routed_expert_gate_up_" + slot, expertHiddenWorker);
                scheduler.addWorkerGrid(prefix + "routed_expert_down_" + slot, dimWorker);
            }
            scheduler.addWorkerGrid(prefix + "shared_expert_gate_up", sharedHiddenWorker);
            scheduler.addWorkerGrid(prefix + "shared_expert_down", dimWorker);
            scheduler.addWorkerGrid(prefix + "shared_expert_gate_and_accumulate", topKWorker);
        }
        return scheduler;
    }

    /** Creates a grid where one GPU work-group computes one output row. */
    private WorkerGrid workerForRows(int rows) {
        WorkerGrid worker = new WorkerGrid1D(rows * LOCAL_WORK_GROUP_SIZE_ALLOC);
        worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);
        return worker;
    }

    /**
     * Creates the complete GPU TaskGraph for one Transformer layer.
     * {@code layerIndex} selects that layer's weights.
     */
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        TaskGraph layer = new TaskGraph("layer_" + layerIndex);
        // Reuse wrapX produced by the previous TaskGraph on the GPU.
        layer.consumeFromDevice(moeState.wrapX);
        // Upload this layer's read-only weights from CPU to GPU on the first execution.
        layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
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
        layer = configureLayerDataTransfers(layer, layerIndex);

        configureAttention(layer, layerIndex);
        configureRoutedExperts(layer, layerIndex);
        layer.persistOnDevice(moeState.wrapX);
        return layer;
    }

    /** Adds the normal Qwen2 attention tasks to this layer's TaskGraph. */
    private void configureAttention(TaskGraph layer, int layerIndex) {
        layer.task("attn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, moeState.temp, moeState.wrapX,
                config.dim(), config.rmsNormEps(), moeState.localSize);

        if (shouldUseFinalNormalization()) {
            layer.task("attn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context, moeState.temp, config.dim(), config.rmsNormEps());
        }

        layer.task("attn_rms_qkv_projection",
                Qwen3Kernels::fusedRmsNormQKVMatmulQ8_0,
                context, moeState.wrapX, moeState.wrapQ, moeState.wrapK, moeState.wrapV,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(), moeState.temp,
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                config.dim(), config.dim(), config.kvDim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        layer.task("fused_qkv_bias",
                TransformerComputeKernelsLayered::fusedQKvBiasAddition,
                context, moeState.wrapQ, moeState.wrapK,
                weights.q_biasLayered[layerIndex].asFloatArray(), moeState.wrapV,
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray(),
                config.dim(), config.kvDim());

        layer.task("rope_and_kv_cache", Qwen3Kernels::ropeRotationWithCacheCopy,
                context, moeState.positionHolder, moeState.wrapQ, moeState.wrapK, moeState.wrapV,
                moeState.wrapKeyCache, moeState.wrapValueCache,
                config.numberOfKeyValueHeads(), config.headSize(), config.kvDim(),
                layerIndex, config.contextLength());

        layer.task("attention", Qwen2Kernels::processHeadsFlashAttention,
                context, moeState.wrapQ, moeState.wrapKeyCache, moeState.wrapValueCache,
                moeState.wrapXb, config.numberOfHeads(), config.headSize(), config.kvDim(),
                config.kvMul(), moeState.positionHolder, layerIndex, config.contextLength());

        layer.task("attn_output_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context, moeState.wrapXb, moeState.wrapX,
                weights.woLayered[layerIndex].asByteArray(),
                config.dim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC);
    }

    /**
     * Adds router, top-K, and selected-expert FFN tasks to this layer's TaskGraph.
     * Their weighted outputs are added to the residual vector.
     */
    private void configureRoutedExperts(TaskGraph layer, int layerIndex) {
        layer.task("ffn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, moeState.tempFFN, moeState.wrapX,
                config.dim(), config.rmsNormEps(), moeState.localSize);

        if (shouldUseFinalNormalization()) {
            layer.task("ffn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context, moeState.tempFFN, config.dim(), config.rmsNormEps());
        }

        layer.task("ffn_rms_apply",
                TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                context, moeState.wrapXb, moeState.wrapX,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), moeState.tempFFN);

        layer.task("router_projection",
                TransformerComputeKernelsLayered::matrixVectorGeneric,
                context, moeState.wrapXb, moeState.wrapRouterLogits,
                weights.routerGateLayered[layerIndex].asFloatArray(),
                config.dim(), config.numberOfExperts(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        // In correctness mode, preserve raw scores before softmax/top-K mutates them.
        if (MoECorrectnessTrace.isEnabled()) {
            layer.task("router_trace_copy", Qwen2MoEKernels::copyRouterLogits,
                    context, moeState.wrapRouterLogits, moeState.wrapRawRouterLogits,
                    config.numberOfExperts());
        }

        layer.task("router_softmax_topk", Qwen2MoEKernels::softmaxAndTopK,
                context, moeState.wrapRouterLogits, moeState.wrapSelectedExperts,
                moeState.wrapRoutingWeights, config.numberOfExperts(), config.numberOfExpertsUsed());

        for (int slot = 0; slot < config.numberOfExpertsUsed(); slot++) {
            layer.task("routed_expert_gate_up_" + slot,
                    Qwen2MoEKernels::fusedRoutedExpertGateUpSwiGLUQ8_0,
                    context, moeState.wrapXb, moeState.wrapSelectedExperts, slot,
                    weights.gateExpertsLayered[layerIndex].asByteArray(),
                    weights.upExpertsLayered[layerIndex].asByteArray(), moeState.wrapExpertGate,
                    config.dim(), config.moeHiddenDim(), config.numberOfExperts(), LOCAL_WORK_GROUP_SIZE_ALLOC);

            layer.task("routed_expert_down_" + slot,
                    Qwen2MoEKernels::routedExpertDownProjectAndAccumulateQ8_0,
                    context, moeState.wrapExpertGate, moeState.wrapX,
                    moeState.wrapSelectedExperts, moeState.wrapRoutingWeights, slot,
                    weights.downExpertsLayered[layerIndex].asByteArray(),
                    config.dim(), config.moeHiddenDim(), config.numberOfExperts(), LOCAL_WORK_GROUP_SIZE_ALLOC);
        }

        // The shared expert always runs; it does not depend on router top-K selection.
        layer.task("shared_expert_gate_up", Qwen2MoEKernels::sharedExpertGateUpSwiGLUQ8_0,
                context, moeState.wrapXb,
                weights.sharedGateLayered[layerIndex].asByteArray(),
                weights.sharedUpLayered[layerIndex].asByteArray(), moeState.wrapSharedGate,
                config.dim(), config.sharedExpertHiddenDim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        layer.task("shared_expert_down", Qwen2MoEKernels::sharedExpertDownProjectQ8_0,
                context, moeState.wrapSharedGate,
                weights.sharedDownLayered[layerIndex].asByteArray(), moeState.wrapSharedOutput,
                config.dim(), config.sharedExpertHiddenDim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        layer.task("shared_expert_gate_and_accumulate", Qwen2MoEKernels::sharedExpertGateAndAccumulate,
                context, moeState.wrapXb, weights.sharedGateInputLayered[layerIndex].asFloatArray(),
                moeState.wrapSharedOutput, moeState.wrapX, config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        if (MoECorrectnessTrace.isEnabled()) {
            layer.transferToHost(DataTransferMode.EVERY_EXECUTION,
                    moeState.wrapRawRouterLogits, moeState.wrapSelectedExperts,
                    moeState.wrapRoutingWeights);
        }
    }

    /**
     * Configures which TaskGraph data is uploaded from the CPU or reused on the GPU.
     */
    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    moeState.positionHolder, moeState.temp, moeState.tempFFN);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context, moeState.wrapXb, moeState.wrapXb2, moeState.wrapQ,
                    moeState.wrapK, moeState.wrapV, moeState.wrapKeyCache,
                    moeState.wrapValueCache, moeState.wrapAtt, moeState.wrapRouterLogits,
                    moeState.wrapSelectedExperts, moeState.wrapRoutingWeights,
                    moeState.wrapExpertGate, moeState.wrapSharedGate, moeState.wrapSharedOutput);
        } else {
            layer.consumeFromDevice(context, moeState.wrapXb, moeState.wrapXb2,
                    moeState.wrapQ, moeState.wrapK, moeState.wrapV, moeState.wrapKeyCache,
                    moeState.wrapValueCache, moeState.wrapAtt, moeState.wrapRouterLogits,
                    moeState.wrapSelectedExperts, moeState.wrapRoutingWeights,
                    moeState.wrapExpertGate, moeState.wrapSharedGate, moeState.wrapSharedOutput,
                    moeState.positionHolder);
        }
        if (MoECorrectnessTrace.isEnabled()) {
            if (layerIndex == 0) {
                layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        moeState.wrapRawRouterLogits);
            } else {
                layer.consumeFromDevice(moeState.wrapRawRouterLogits);
            }
        }
        return layer;
    }
}
