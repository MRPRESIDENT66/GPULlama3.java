package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

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

    public Qwen2MoEState(Configuration config, int batchsize) {
        super(config, batchsize);  // allocate all the regular Qwen2State buffers first
        Qwen2MoEConfiguration c = (Qwen2MoEConfiguration) config;
        this.routerLogits = ArrayFloatTensor.allocate(c.numberOfExperts());
        this.hbE  = ArrayFloatTensor.allocate(c.moeHiddenDim());
        this.hbE2 = ArrayFloatTensor.allocate(c.moeHiddenDim());
        this.hbS  = ArrayFloatTensor.allocate(c.sharedExpertHiddenDim());
        this.hbS2 = ArrayFloatTensor.allocate(c.sharedExpertHiddenDim());
        this.yTmp = ArrayFloatTensor.allocate(c.dim());
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

        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), nEmbdGqa)).limit(config.numberOfLayers()).toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), nEmbdGqa)).limit(config.numberOfLayers()).toArray(FloatTensor[]::new);

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

        fields.temp = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));
        fields.tempFFN = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));
        fields.tempLogits = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));

        return fields;
    }
}
