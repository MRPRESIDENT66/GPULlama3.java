package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;

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
    protected StateFields createStateFields(Configuration config) {
        return super.createStateFields(config);
    }
}
