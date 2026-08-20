package org.beehive.gpullama3.inference.state;

import java.util.stream.Stream;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

public class Qwen2MoEState extends Qwen2State {

    // Router output scores, one per expert (length = numberOfExperts).
    // Used to pick the top-k experts for the current token.
    public final FloatTensor routerLogits;

    // Scratch buffers for a single routed expert's internal FFN (length = moeHiddenDim).
    // hbE holds gate_proj(xb), hbE2 holds up_proj(xb); combined via silu(hbE) * hbE2.
    public final FloatTensor hbE;
    public final FloatTensor hbE2;

    // Scratch buffers for the shared expert's internal FFN (length = sharedExpertHiddenDim).
    // Same role as hbE/hbE2, but for the always-on shared expert instead of a routed one.
    public final FloatTensor hbS;
    public final FloatTensor hbS2;

    // Temporary holder for a single expert's down-projected output (length = dim),
    // before it is weighted and accumulated into the residual stream (state.x).
    public final FloatTensor yTmp;

    // TornadoVM buffers for the single-token GPU MoE path.  These are deliberately
    // separate from the CPU FloatTensor fields above: TaskGraph kernels operate on
    // TornadoVM arrays that can remain resident on the device between tasks.
    public final FloatArray wrapRouterLogits;
    public final IntArray wrapSelectedExperts;
    public final FloatArray wrapRoutingWeights;
    public final FloatArray wrapExpertGate;
    public final FloatArray wrapSharedGate;
    public final FloatArray wrapSharedOutput;

    // TornadoVM buffers for the batch-prefill MoE path.
    // Their shapes use the configured maximum batch size so TaskGraphs stay fixed.
    public final FloatArray wrapRouterLogitsBatch;
    public final IntArray activeBatchSizeHolder;
    public final IntArray wrapSelectedExpertsBatch;
    public final FloatArray wrapRoutingWeightsBatch;
    public final IntArray wrapGroupedAssignmentIds;
    public final IntArray wrapGroupedPositionByAssignment;
    public final IntArray wrapExpertTileIds;
    public final IntArray wrapExpertTileStarts;
    public final IntArray wrapExpertTileCounts;
    public final IntArray wrapExpertTileCountHolder;
    public final FloatArray wrapGroupedExpertHidden;
    public final FloatArray wrapGroupedExpertDown;
    public final FloatArray wrapSharedHiddenBatch;
    public final FloatArray wrapSharedWeightBatch;

    // Experimental W8A8 buffers for routed Gate/Up during batch prefill.
    // The normalized FFN input is quantized once per token and Q8_0 block, then
    // reused by every routed-expert output row in the layer.
    public final ByteArray wrapQuantizedFfnInputBatch;
    public final HalfFloatArray wrapQuantizedFfnInputScalesBatch;
    public final FloatArray wrapDebugFp32ExpertHidden;

    public Qwen2MoEState(Configuration config, int batchsize) {
        super(config, batchsize);
        Qwen2MoEConfiguration c = (Qwen2MoEConfiguration) config;
        this.routerLogits = ArrayFloatTensor.allocate(c.numberOfExperts());
        this.hbE = ArrayFloatTensor.allocate(c.moeHiddenDim());
        this.hbE2 = ArrayFloatTensor.allocate(c.moeHiddenDim());
        this.hbS = ArrayFloatTensor.allocate(c.sharedExpertHiddenDim());
        this.hbS2 = ArrayFloatTensor.allocate(c.sharedExpertHiddenDim());
        this.yTmp = ArrayFloatTensor.allocate(c.dim());

        this.wrapRouterLogits = new FloatArray(c.numberOfExperts());
        this.wrapSelectedExperts = new IntArray(c.numberOfExpertsUsed());
        this.wrapRoutingWeights = new FloatArray(c.numberOfExpertsUsed());
        this.wrapExpertGate = new FloatArray(c.moeHiddenDim() * c.numberOfExpertsUsed());
        this.wrapSharedGate = new FloatArray(c.sharedExpertHiddenDim());
        this.wrapSharedOutput = new FloatArray(c.dim());

        int gpuBatchSize = Integer.getInteger("llama.prefillBatchSize", 1);
        if (gpuBatchSize > 1) {
            int assignments = gpuBatchSize * c.numberOfExpertsUsed();
            this.wrapRouterLogitsBatch = new FloatArray(gpuBatchSize * c.numberOfExperts());
            this.activeBatchSizeHolder = new IntArray(1);
            this.activeBatchSizeHolder.init(gpuBatchSize);
            this.wrapSelectedExpertsBatch = new IntArray(assignments);
            this.wrapRoutingWeightsBatch = new FloatArray(assignments);
            this.wrapGroupedAssignmentIds = new IntArray(assignments);
            this.wrapGroupedPositionByAssignment = new IntArray(assignments);
            this.wrapExpertTileIds = new IntArray(assignments);
            this.wrapExpertTileStarts = new IntArray(assignments);
            this.wrapExpertTileCounts = new IntArray(assignments);
            this.wrapExpertTileCountHolder = new IntArray(1);
            this.wrapGroupedExpertHidden = new FloatArray(assignments * c.moeHiddenDim());
            this.wrapGroupedExpertDown = new FloatArray(assignments * c.dim());
            this.wrapSharedHiddenBatch = new FloatArray(gpuBatchSize * c.sharedExpertHiddenDim());
            this.wrapSharedWeightBatch = new FloatArray(gpuBatchSize);
            int blocksPerInput = (c.dim() + 31) / 32;
            this.wrapQuantizedFfnInputBatch = new ByteArray(gpuBatchSize * c.dim());
            this.wrapQuantizedFfnInputScalesBatch =
                    new HalfFloatArray(gpuBatchSize * blocksPerInput);
            this.wrapDebugFp32ExpertHidden = new FloatArray(assignments * c.moeHiddenDim());
        } else {
            this.wrapRouterLogitsBatch = null;
            this.activeBatchSizeHolder = null;
            this.wrapSelectedExpertsBatch = null;
            this.wrapRoutingWeightsBatch = null;
            this.wrapGroupedAssignmentIds = null;
            this.wrapGroupedPositionByAssignment = null;
            this.wrapExpertTileIds = null;
            this.wrapExpertTileStarts = null;
            this.wrapExpertTileCounts = null;
            this.wrapExpertTileCountHolder = null;
            this.wrapGroupedExpertHidden = null;
            this.wrapGroupedExpertDown = null;
            this.wrapSharedHiddenBatch = null;
            this.wrapSharedWeightBatch = null;
            this.wrapQuantizedFfnInputBatch = null;
            this.wrapQuantizedFfnInputScalesBatch = null;
            this.wrapDebugFp32ExpertHidden = null;
        }
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        StateFields fields = new StateFields();

        Qwen2MoEConfiguration config = (Qwen2MoEConfiguration) configuration;

        int nEmbdGqa = config.kvDim();

        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.xb = ArrayFloatTensor.allocate(config.dim());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(config.dim());
        fields.k = ArrayFloatTensor.allocate(config.kvDim());
        fields.v = ArrayFloatTensor.allocate(config.kvDim());
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), nEmbdGqa))
                        .limit(config.numberOfLayers())
                        .toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), nEmbdGqa))
                        .limit(config.numberOfLayers())
                        .toArray(FloatTensor[]::new);

        switch (config.quantization()) {
            case "FP16" -> fields.createActivationFP16(config.dim());
            case "Q8_0" -> fields.createActivationQ8_0(config.dim());
            default -> throw new UnsupportedOperationException("Unsupported quantization format: " + config.quantization());
        }
        fields.wrapX = new FloatArray(config.dim());
        fields.wrapXb = new FloatArray(config.dim());
        fields.wrapXbFP16 = new HalfFloatArray(config.dim());
        fields.wrapXb2 = new FloatArray(config.dim());
        fields.wrapHb = new FloatArray(config.hiddenDim());
        fields.wrapHb2 = new FloatArray(config.hiddenDim());

        fields.wrapLogits = new FloatArray(config.vocabularySize());
        fields.wrapQ = new FloatArray(config.dim());
        fields.wrapK = new FloatArray(config.kvDim());
        fields.wrapV = new FloatArray(config.kvDim());

        fields.wrapKeyCache = new FloatArray(config.contextLength() * nEmbdGqa * config.numberOfLayers());
        fields.wrapValueCache = new FloatArray(config.contextLength() * nEmbdGqa * config.numberOfLayers());
        fields.wrapValueCache.init(0.f);
        fields.wrapKeyCache.init(0.f);
        fields.wrapAtt = new FloatArray(config.numberOfHeads() * config.contextLength());
        fields.positionHolder = new IntArray(1);

        // State invokes this override before the Qwen2State constructor body runs,
        // so use the Qwen2 work-group size directly instead of State.localSize.
        fields.temp = new FloatArray(1 + ((config.dim() + QWEN2_LOCAL_SIZE - 1) / QWEN2_LOCAL_SIZE));
        fields.tempFFN = new FloatArray(1 + ((config.dim() + QWEN2_LOCAL_SIZE - 1) / QWEN2_LOCAL_SIZE));
        fields.tempLogits = new FloatArray(1 + ((config.dim() + QWEN2_LOCAL_SIZE - 1) / QWEN2_LOCAL_SIZE));

        return fields;
    }
}
