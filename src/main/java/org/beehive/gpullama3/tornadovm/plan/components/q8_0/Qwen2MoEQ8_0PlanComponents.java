package org.beehive.gpullama3.tornadovm.plan.components.q8_0;

import org.beehive.gpullama3.inference.state.Qwen2MoEState;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2MoETornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.Qwen2MoEQ8_0FFNLayers;
import org.beehive.gpullama3.tornadovm.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;

/** Assembles the single-token Q8_0 GPU components for Qwen2-MoE. */
public final class Qwen2MoEQ8_0PlanComponents implements SingleTokenForwardPlanComponents {

    private final Qwen2MoEState state;
    private final Qwen2MoETornadoWeights weights;
    private final Qwen2MoEConfiguration config;
    private final SchedulerType schedulerType;

    public Qwen2MoEQ8_0PlanComponents(Qwen2MoEState state, Model model) {
        this.state = state;
        this.config = (Qwen2MoEConfiguration) model.configuration();
        this.weights = (Qwen2MoETornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Qwen2MoEQ8_0FFNLayers("qwen2MoEFFN", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsQ8_0Layer("logits", state, weights, config, previousGraphId, schedulerType);
    }
}
